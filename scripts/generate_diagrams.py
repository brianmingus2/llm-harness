#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = BASE_DIR / "diagrams"
PYTHON_SOURCES = [
    BASE_DIR / "harness.py",
    BASE_DIR / "scripts" / "ui_playwright_check.py",
    BASE_DIR / "tools" / "github_bootstrap.py",
    BASE_DIR / "rxconfig.py",
]


def _run(cmd: list[str], cwd: Path = BASE_DIR) -> bool:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        suffix = f": {msg}" if msg else ""
        sys.stderr.write(f"warn: {' '.join(cmd)}{suffix}\n")
        return False
    return True


def _render_dot(dot_path: Path, output_path: Path) -> bool:
    if not dot_path.exists():
        sys.stderr.write(f"warn: missing dot file {dot_path}\n")
        return False
    fmt = output_path.suffix.lstrip(".") or "svg"
    return _run(
        ["dot", f"-T{fmt}", str(dot_path), "-o", str(output_path)],
        cwd=dot_path.parent,
    )


def _render_run_all_tests_flow() -> bool:
    try:
        from graphviz import Digraph
    except Exception as exc:
        sys.stderr.write(f"warn: graphviz import failed: {exc}\n")
        return False
    dot = Digraph("run_all_tests", graph_attr={"rankdir": "LR"})
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#e8f1ff",
        fontname="Helvetica",
    )
    dot.node("start", "Start", shape="oval", fillcolor="#ffffff")
    dot.node("maxcov", "Maximum Coverage")
    dot.node("static", "Static Analysis Suite")
    dot.node("reflex", "Reflex Run (clean)")
    dot.node("ui", "UI Playwright Check")
    dot.node("summary", "Summary Table", shape="oval", fillcolor="#ffffff")
    dot.edges(
        [
            ("start", "maxcov"),
            ("maxcov", "static"),
            ("static", "reflex"),
            ("reflex", "ui"),
            ("ui", "summary"),
        ]
    )
    output = DIAGRAMS_DIR / "run_all_tests_flow"
    dot.format = "svg"
    dot.render(str(output), cleanup=True)
    return True


def _render_architecture_diagram() -> bool:
    try:
        from diagrams import Diagram
        from diagrams.generic.storage import Storage
        from diagrams.onprem.client import User
        from diagrams.onprem.compute import Server
        from diagrams.onprem.database import Neo4J
        from diagrams.programming.language import Python
    except Exception as exc:
        sys.stderr.write(f"warn: diagrams import failed: {exc}\n")
        return False
    output = DIAGRAMS_DIR / "resume_builder_architecture"
    with Diagram(
        "Resume Builder Architecture",
        filename=str(output),
        outformat="svg",
        show=False,
    ):
        user = User("User")
        harness = Python("Harness CLI")
        playwright = Python("Playwright")
        ui = Server("Reflex UI")
        db = Neo4J("Neo4j")
        llm = Python("LLM")
        typst = Python("Typst")
        pdf = Storage("PDF")
        user >> ui
        harness >> ui
        playwright >> ui
        ui >> db
        ui >> llm
        ui >> typst >> pdf
    return True


def _render_pipeline_diagram() -> bool:
    try:
        import pydot
    except Exception as exc:
        sys.stderr.write(f"warn: pydot import failed: {exc}\n")
        return False
    graph = pydot.Dot(
        "resume_pipeline",
        graph_type="digraph",
        rankdir="LR",
        fontname="Helvetica",
    )
    graph.set_node_defaults(shape="box", style="rounded,filled", fillcolor="#eef7ff")
    nodes = {
        "req": "Job Req",
        "prompt": "Prompt Template",
        "llm": "LLM",
        "profile": "Profile Data",
        "neo4j": "Neo4j",
        "typst": "Typst",
        "pdf": "PDF",
    }
    for key, label in nodes.items():
        graph.add_node(pydot.Node(key, label=label))
    for src, dst in [
        ("req", "prompt"),
        ("prompt", "llm"),
        ("llm", "profile"),
        ("profile", "neo4j"),
        ("neo4j", "typst"),
        ("typst", "pdf"),
    ]:
        graph.add_edge(pydot.Edge(src, dst))
    output = DIAGRAMS_DIR / "resume_pipeline.svg"
    graph.write_svg(str(output))
    return output.exists()


def _render_pyreverse() -> bool:
    if shutil.which("pyreverse") is None:
        sys.stderr.write("warn: pyreverse missing\n")
        return False
    sources = [str(path) for path in PYTHON_SOURCES if path.exists()]
    with tempfile.TemporaryDirectory(prefix="pyreverse_") as tmpdir:
        tmp_path = Path(tmpdir)
        cmd = [
            "pyreverse",
            "-o",
            "dot",
            "-p",
            "resume_builder",
            "-d",
            str(tmp_path),
            *sources,
        ]
        if not _run(cmd):
            return False
        class_candidates = sorted(tmp_path.glob("classes_*.dot"))
        package_candidates = sorted(tmp_path.glob("packages_*.dot"))
        ok = True
        if class_candidates:
            dot_path = class_candidates[0]
            shutil.copy2(dot_path, DIAGRAMS_DIR / "pyreverse_classes.dot")
            ok = _render_dot(dot_path, DIAGRAMS_DIR / "pyreverse_classes.svg") and ok
        else:
            ok = False
        if package_candidates:
            dot_path = package_candidates[0]
            shutil.copy2(dot_path, DIAGRAMS_DIR / "pyreverse_packages.dot")
            ok = _render_dot(dot_path, DIAGRAMS_DIR / "pyreverse_packages.svg") and ok
        else:
            ok = False
        return ok


def _render_pydeps() -> bool:
    if shutil.which("pydeps") is None:
        sys.stderr.write("warn: pydeps missing\n")
        return False
    output = DIAGRAMS_DIR / "pydeps.svg"
    cmd = [
        "pydeps",
        str(BASE_DIR / "harness.py"),
        "-T",
        "svg",
        "-o",
        str(output),
        "--max-module-depth",
        "2",
        "--cluster",
        "--rankdir",
        "LR",
        "--noshow",
    ]
    return _run(cmd)


def _render_pyan() -> bool:
    try:
        import logging
        from pyan import analyzer, visgraph, writers
    except Exception as exc:
        sys.stderr.write(f"warn: pyan import failed: {exc}\n")
        return False
    sources = [str(path) for path in PYTHON_SOURCES if path.exists()]
    if not sources:
        return False
    logger = logging.getLogger("pyan")
    logger.setLevel(logging.WARNING)
    try:
        visitor = analyzer.CallGraphVisitor(sources, logger=logger)
    except Exception as exc:
        sys.stderr.write(f"warn: pyan failed on full sources: {exc}\n")
        fallback = BASE_DIR / "rxconfig.py"
        if not fallback.exists():
            return False
        visitor = analyzer.CallGraphVisitor([str(fallback)], logger=logger)
    options = {
        "colored": False,
        "draw_defines": True,
        "draw_uses": True,
        "grouped": False,
        "nested_groups": False,
        "grouped_alt": False,
        "annotated": False,
    }
    graph = visgraph.VisualGraph.from_visitor(visitor, options=options, logger=logger)
    dot_path = DIAGRAMS_DIR / "call_graph.dot"
    writer = writers.DotWriter(graph, output=str(dot_path), logger=logger)
    writer.run()
    return _render_dot(dot_path, DIAGRAMS_DIR / "call_graph.svg")


def main() -> int:
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("run_all_tests_flow", _render_run_all_tests_flow),
        ("architecture", _render_architecture_diagram),
        ("pipeline", _render_pipeline_diagram),
        ("pyreverse", _render_pyreverse),
        ("pydeps", _render_pydeps),
        ("pyan3", _render_pyan),
    ]
    failures = 0
    for label, func in tasks:
        if not func():
            sys.stderr.write(f"warn: diagram step failed: {label}\n")
            failures += 1
    if failures:
        sys.stderr.write(f"warn: {failures} diagram step(s) failed\n")
        return 1
    print(f"diagrams written to {DIAGRAMS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
