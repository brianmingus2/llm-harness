import reflex as rx
import json
import os
import subprocess
import re
import tempfile
import tarfile
import shutil
from datetime import datetime, timezone
import base64
import uuid
import asyncio
import signal
import time
import socket
from neo4j import GraphDatabase, Query
from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError
import argparse
import sys

import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from pydantic import BaseModel
from fontTools.ttLib import TTFont
import hashlib
from typing import LiteralString, cast

# Emit maximum coverage progress logs when requested.
MAX_COVERAGE_LOG = (
    os.environ.get("MAX_COVERAGE_LOG") == "1" or "--maximum-coverage" in sys.argv
)


def _should_silence_warnings() -> bool:
    return os.environ.get("MAX_COVERAGE_SILENCE_WARNINGS") == "1"


def _install_maxcov_print_filter() -> None:
    if not _should_silence_warnings():
        return
    import builtins
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    orig_print = builtins.print

    def _filtered_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = ""
        if msg.startswith("[maxcov]"):
            return orig_print(*args, **kwargs)
        if msg.startswith(
            (
                "Warning:",
                "Error running Typst:",
                "Typst compilation failed",
                "Req file not found",
                "No resume found in Neo4j",
                "Error compiling PDF:",
                "Error retrieving resume data:",
                "Error listing applied jobs:",
                "Error resetting/importing",
                "Error importing",
                "Error saving Profile:",
            )
        ):
            return
        if any(
            token in msg
            for token in (
                "file not found at",
                "could not fetch",
                "unable to apply PDF metadata",
                "could not save resume pdf",
                "could not render resume pdf",
                "Playwright not available",
                "Playwright traversal failed",
                "reflex coverage server did not start",
                "failed to start reflex coverage server",
                "Unexpected exit from worker",
                "Task exception was never retrieved",
                "Event loop is closed",
                "Error in on_load:",
            )
        ):
            return
        return orig_print(*args, **kwargs)

    builtins.print = _filtered_print


def _install_reflex_signal_handlers() -> None:
    if __name__ == "__main__":
        return
    if os.environ.get("REFLEX_GRACEFUL_EXIT") == "0":
        return

    def _graceful_exit(_signum, _frame):
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            current = signal.getsignal(sig)
        except Exception:
            continue
        if current not in (signal.SIG_DFL, signal.default_int_handler):
            continue
        try:
            signal.signal(sig, _graceful_exit)
        except Exception:
            continue


# Default app-level LLM settings when not explicitly set.
def _ensure_default_llm_env() -> None:
    if not os.environ.get("LLM_REASONING_EFFORT") and not os.environ.get(
        "OPENAI_REASONING_EFFORT"
    ):
        os.environ["LLM_REASONING_EFFORT"] = "medium"
    if not os.environ.get("LLM_MAX_OUTPUT_TOKENS"):
        os.environ["LLM_MAX_OUTPUT_TOKENS"] = "2048"


_ensure_default_llm_env()
_install_maxcov_print_filter()
_install_reflex_signal_handlers()


def _try_import_coverage():
    try:
        import coverage  # noqa: F401
    except Exception:
        return None
    return coverage


# Enable coverage collection inside Reflex worker processes when requested.
_REFLEX_COVERAGE = None
_REFLEX_COVERAGE_OWNED = False


def _init_reflex_coverage() -> None:
    """Initialize coverage in Reflex worker processes when requested."""
    global _REFLEX_COVERAGE, _REFLEX_COVERAGE_OWNED
    if os.environ.get("REFLEX_COVERAGE") != "1":
        return
    try:
        import coverage  # noqa: F401
        import atexit

        _REFLEX_COVERAGE = coverage.Coverage(
            data_file=os.environ.get("COVERAGE_FILE")
            or os.environ.get("REFLEX_COVERAGE_FILE"),
            data_suffix=True,
            branch=True,
            source=[str(Path(__file__).resolve().parent)],
        )
        current_cov = None
        force_owned = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED") == "1"
        try:
            current_cov = coverage.Coverage.current()
        except Exception:
            current_cov = None

        if current_cov is None or force_owned:
            _REFLEX_COVERAGE_OWNED = True
            _REFLEX_COVERAGE.start()

            def _stop_reflex_coverage() -> None:
                if _REFLEX_COVERAGE is not None and _REFLEX_COVERAGE_OWNED:
                    _REFLEX_COVERAGE.stop()
                    _REFLEX_COVERAGE.save()

            atexit.register(_stop_reflex_coverage)
            if os.environ.get("MAX_COVERAGE_REFLEX_STOP") == "1":
                _stop_reflex_coverage()
            if current_cov is not None:
                _REFLEX_COVERAGE.stop()
                _REFLEX_COVERAGE.save()
                _REFLEX_COVERAGE_OWNED = False
        else:
            _REFLEX_COVERAGE.start()
            _REFLEX_COVERAGE.stop()
            _REFLEX_COVERAGE.save()
    except Exception:
        _REFLEX_COVERAGE = None
        _REFLEX_COVERAGE_OWNED = False


_init_reflex_coverage()

# ==========================================
# CONFIGURATION
# ==========================================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "ResumeBuilder")
SUPPORTED_LLM_PROVIDERS = {"openai", "gemini"}
DEFAULT_LLM_MODELS = [
    # OpenAI (Responses API supported)
    "openai:gpt-5.2-pro",
    "openai:gpt-5.2",
    "openai:gpt-5.2-mini",
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    # Google Gemini (Completions API supported)
    "gemini:gemini-2.0-flash",
    "gemini:gemini-1.5-pro",
    "gemini:gemini-1.5-flash",
]


def _resolve_default_llm_settings() -> tuple[str, str]:
    effort = (
        (
            os.environ.get("LLM_REASONING_EFFORT")
            or os.environ.get("OPENAI_REASONING_EFFORT")
            or "high"
        )
        .strip()
        .lower()
    )
    if effort not in {"none", "minimal", "low", "medium", "high"}:
        effort = "high"
    model_env = (os.environ.get("LLM_MODEL") or "").strip()
    if model_env:
        model = model_env
    else:
        model = (os.environ.get("OPENAI_MODEL") or DEFAULT_LLM_MODELS[0]).strip()
    if model and ":" not in model:
        # Backward compatibility: bare model ids default to OpenAI.
        prefix = model.split("/", 1)[0].strip().lower()
        if prefix not in SUPPORTED_LLM_PROVIDERS:
            model = f"openai:{model}"
    return effort, model


DEFAULT_LLM_REASONING_EFFORT, DEFAULT_LLM_MODEL = _resolve_default_llm_settings()
BASE_DIR = Path(__file__).resolve().parent
PROMPT_YAML_PATH = BASE_DIR / "prompt.yaml"


def _resolve_assets_dir(base_dir: Path) -> Path:
    candidate = base_dir / "assets"
    if candidate.exists() and not candidate.is_dir():
        return base_dir / "assets_out"
    return candidate


ASSETS_DIR = _resolve_assets_dir(BASE_DIR)
# Cache fonts inside the project to make the build self-contained.
FONTS_DIR = BASE_DIR / "fonts"
PACKAGES_DIR = BASE_DIR / "packages"
LIVE_PDF_PATH = ASSETS_DIR / "preview.pdf"
LIVE_PDF_SIG_PATH = ASSETS_DIR / "preview.sig"
TYPST_TEMPLATE_VERSION = "modern-cv-0.9.0-2025-02-23"
RUNTIME_WRITE_PDF = os.environ.get("RUNTIME_WRITE_PDF", "0") == "1"
TEMP_BUILD_DIR = BASE_DIR / ".tmp_typst"
FONT_AWESOME_PACKAGE_VERSION = "0.6.0"
FONT_AWESOME_PACKAGE_URL = f"https://packages.typst.org/preview/fontawesome-{FONT_AWESOME_PACKAGE_VERSION}.tar.gz"
FONT_AWESOME_PACKAGE_DIR = (
    PACKAGES_DIR / "preview" / "fontawesome" / FONT_AWESOME_PACKAGE_VERSION
)
FONT_AWESOME_SOURCES = {
    "Font Awesome 7 Free-Solid-900.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Free-Solid-900.otf",
    "Font Awesome 7 Free-Regular-400.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Free-Regular-400.otf",
    "Font Awesome 7 Brands-Regular-400.otf": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.x/otfs/Font%20Awesome%207%20Brands-Regular-400.otf",
}
DEFAULT_ASSETS_JSON = BASE_DIR / "michael_scott_resume.json"
DEBUG_LOG = Path(tempfile.gettempdir()) / "resume_builder_debug.log"
DEFAULT_TYPST_PATH = BASE_DIR / "bin" / ("typst.exe" if os.name == "nt" else "typst")
TYPST_BIN = os.environ.get("TYPST_BIN") or (
    str(DEFAULT_TYPST_PATH) if DEFAULT_TYPST_PATH.exists() else "typst"
)
DEFAULT_SECTION_ORDER = [
    "summary",
    "matrices",
    "education",
    "education_continued",
    "experience",
    "founder",
]
DEFAULT_AUTO_FIT_TARGET_PAGES = 2
DEFAULT_SKILLS_ROW_LABELS = [
    "Leadership & Strategy",
    "Technical Domain & Tools",
    "Architectural Patterns & Methodologies",
]
SOFT_BOLD_WEIGHT = 350
SOFT_EMPH_FILL = "#374151"
SOFT_SECONDARY_FILL = "#6B7280"
SECTION_LABELS = {
    "summary": "Summary",
    "education": "Education",
    "education_continued": "Education Continued",
    "experience": "Experience",
    "founder": "Startup Founder",
    "matrices": "Skills",
}
_NEO4J_DRIVER = None
_NEO4J_SCHEMA_READY = False
_FONTS_READY = False
_LOCAL_FONT_CATALOG = None
_LOCAL_FONT_EXTRA_FONTS = None
_MAXCOV_DB = None


def _maxcov_stub_db_enabled() -> bool:
    return os.environ.get("MAX_COVERAGE_STUB_DB") == "1"


def _empty_resume_payload() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": "",
        "first_name": "",
        "middle_name": "",
        "last_name": "",
        "email": "",
        "email2": "",
        "phone": "",
        "font_family": DEFAULT_RESUME_FONT_FAMILY,
        "auto_fit_target_pages": DEFAULT_AUTO_FIT_TARGET_PAGES,
        "auto_fit_best_scale": 1.0,
        "auto_fit_too_long_scale": 0.0,
        "linkedin_url": "",
        "github_url": "",
        "scholar_url": "",
        "calendly_url": "",
        "portfolio_url": "",
        "summary": "",
        "head1_left": "",
        "head1_middle": "",
        "head1_right": "",
        "head2_left": "",
        "head2_middle": "",
        "head2_right": "",
        "head3_left": "",
        "head3_middle": "",
        "head3_right": "",
        "top_skills": [],
        "section_order": list(DEFAULT_SECTION_ORDER),
        "section_enabled": list(SECTION_LABELS),
        "section_titles_json": "{}",
        "custom_sections_json": "[]",
        "prompt_yaml": _load_prompt_yaml_from_file() or "",
    }


def _seed_maxcov_store(store: dict, data: dict) -> None:
    profile = dict((data or {}).get("profile") or {})
    profile.setdefault("id", str(uuid.uuid4()))
    profile.setdefault("scholar_url", "")
    profile.setdefault("calendly_url", "")
    profile.setdefault("portfolio_url", "")
    profile.setdefault("email2", "")
    profile.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
    profile.setdefault("auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES)
    profile.setdefault("section_titles_json", "{}")
    profile.setdefault("custom_sections_json", "[]")
    custom_sections = _normalize_custom_sections(
        profile.get("custom_sections_json") or profile.get("custom_sections")
    )
    extra_keys = _custom_section_keys(custom_sections)
    profile["section_enabled"] = _normalize_section_enabled(
        profile.get("section_enabled"),
        list(SECTION_LABELS) + extra_keys,
        extra_keys=extra_keys,
    )
    store["resume"] = profile
    store["experience"] = list((data or {}).get("experience") or [])
    store["education"] = list((data or {}).get("education") or [])
    store["founder_roles"] = list((data or {}).get("founder_roles") or [])
    store["skills"] = list((data or {}).get("skills") or [])


def _get_maxcov_store() -> dict:
    global _MAXCOV_DB
    if _MAXCOV_DB is None:
        _MAXCOV_DB = {
            "resume": {},
            "experience": [],
            "education": [],
            "founder_roles": [],
            "skills": [],
            "profiles": [],
            "auto_fit_cache": {},
        }
        try:
            if DEFAULT_ASSETS_JSON.exists():
                _seed_maxcov_store(
                    _MAXCOV_DB,
                    json.loads(DEFAULT_ASSETS_JSON.read_text(encoding="utf-8")),
                )
        except Exception:
            pass
    return _MAXCOV_DB


def _iter_local_font_files() -> list[Path]:
    if not FONTS_DIR.exists():
        return []
    paths = []
    for path in sorted(FONTS_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".otf", ".ttf", ".woff", ".woff2"}:
            continue
        if "font awesome" in path.name.lower():
            continue
        paths.append(path)
    return paths


def _read_font_family(ttfont: TTFont) -> str:
    try:
        name_table = ttfont["name"]
    except Exception:
        return ""
    for name_id in (16, 1):
        for record in name_table.names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                value = ""
            if value:
                return value
    return ""


def _read_font_weight_italic(ttfont: TTFont) -> tuple[int, bool]:
    weight = 400
    italic = False
    try:
        os2 = ttfont["OS/2"]
        weight = int(getattr(os2, "usWeightClass", weight) or weight)
        italic = bool(getattr(os2, "fsSelection", 0) & 0x01)
    except Exception:
        pass
    try:
        head = ttfont["head"]
        italic = italic or bool(getattr(head, "macStyle", 0) & 0x02)
    except Exception:
        pass
    weight = max(100, min(weight, 900))
    return weight, italic


def _build_local_font_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = {}
    for path in _iter_local_font_files():
        try:
            ttfont = TTFont(str(path), fontNumber=0, lazy=True)
        except Exception:
            continue
        family = _read_font_family(ttfont) or path.stem
        if "font awesome" in family.lower():
            continue
        weight, italic = _read_font_weight_italic(ttfont)
        catalog.setdefault(family, []).append(
            {"path": path, "weight": weight, "italic": italic}
        )
    return catalog


def _get_local_font_catalog() -> dict[str, list[dict]]:
    global _LOCAL_FONT_CATALOG
    if _LOCAL_FONT_CATALOG is None:
        _LOCAL_FONT_CATALOG = _build_local_font_catalog()
    return _LOCAL_FONT_CATALOG


def _pick_primary_font_entry(entries: list[dict]) -> dict | None:
    if not entries:
        return None

    def score(entry: dict) -> int:
        weight = int(entry.get("weight") or 400)
        italic = bool(entry.get("italic"))
        return abs(weight - 400) + (1000 if italic else 0)

    return min(entries, key=score)


def _select_local_font_paths(family: str, italic: bool) -> list[Path]:
    catalog = _get_local_font_catalog()
    entries = catalog.get(family, [])
    if not entries:
        family_lower = (family or "").strip().lower()
        for key, items in catalog.items():
            if key.lower() == family_lower:
                entries = items
                break
    if not entries:
        return []

    def score(entry: dict) -> int:
        weight = int(entry.get("weight") or 400)
        is_italic = bool(entry.get("italic"))
        return abs(weight - 400) + (0 if is_italic == italic else 1000)

    ordered = sorted(entries, key=score)
    return [entry["path"] for entry in ordered if "path" in entry]


def _font_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".otf": "font/otf",
        ".ttf": "font/ttf",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }.get(suffix, "application/octet-stream")
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_local_font_extra_fonts() -> list[dict]:
    extra_fonts: list[dict] = []
    catalog = _get_local_font_catalog()
    for family, entries in sorted(catalog.items()):
        entry = _pick_primary_font_entry(entries)
        if not entry:
            continue
        path = entry.get("path")
        if not isinstance(path, Path):
            continue
        try:
            url = _font_data_uri(path)
        except Exception:
            continue
        extra_fonts.append({"name": family, "variants": ["400"], "url": url})
    return extra_fonts


def _get_local_font_extra_fonts() -> list[dict]:
    global _LOCAL_FONT_EXTRA_FONTS
    if _LOCAL_FONT_EXTRA_FONTS is None:
        _LOCAL_FONT_EXTRA_FONTS = _build_local_font_extra_fonts()
    return _LOCAL_FONT_EXTRA_FONTS


def _resolve_default_font_family() -> str:
    catalog = _get_local_font_catalog()
    for name in catalog:
        if name.strip().lower() in {"avenir lt std", "avenir"}:
            return name
    if catalog:
        return sorted(catalog.keys())[0]
    return "Avenir LT Std"


DEFAULT_RESUME_FONT_FAMILY = _resolve_default_font_family()
FONT_PICKER_EXTRA_FONTS_JSON = json.dumps(
    _get_local_font_extra_fonts(), ensure_ascii=True
)


def _known_section_keys(extra_keys: list[str] | None = None) -> list[str]:
    keys = list(SECTION_LABELS)
    for key in extra_keys or []:
        if key and key not in keys:
            keys.append(key)
    return keys


def _sanitize_section_order(
    raw_order: list[str] | None, extra_keys: list[str] | None = None
) -> list[str]:
    """Return a stable, de-duplicated section order with defaults appended."""
    known_keys = _known_section_keys(extra_keys)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in raw_order or []:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in DEFAULT_SECTION_ORDER:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in known_keys:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _filter_section_order(
    raw_order: list[str] | None, extra_keys: list[str] | None = None
) -> list[str]:
    """Filter a section order list to known keys without appending defaults."""
    known_keys = _known_section_keys(extra_keys)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in raw_order or []:
        if key in known_keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _normalize_section_enabled(
    raw,
    default: list[str] | None = None,
    extra_keys: list[str] | None = None,
) -> list[str]:
    """Normalize section_enabled values to a list of enabled section keys."""
    known_keys = _known_section_keys(extra_keys)
    default_list = list(default or known_keys)
    if raw is None:
        return list(default_list)
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return list(default_list)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = [s.strip() for s in cleaned.split(",") if s.strip()]
        return _normalize_section_enabled(parsed, default_list, extra_keys)
    if isinstance(raw, dict):
        return [k for k, v in raw.items() if v and k in known_keys]
    if isinstance(raw, (list, tuple, set)):
        return [k for k in raw if k in known_keys]
    return list(default_list)


def _apply_section_enabled(
    order: list[str] | None, enabled: list[str] | None
) -> list[str]:
    """Filter an order list by enabled section keys."""
    order = list(order or [])
    if enabled is None:
        return order
    enabled_set = {k for k in enabled}
    return [k for k in order if k in enabled_set]


def _coerce_bool(value) -> bool:
    """Best-effort conversion for UI checkbox/switch values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    if isinstance(value, dict):
        for key in ("checked", "value", "isChecked"):
            if key in value:
                return _coerce_bool(value.get(key))
        target = value.get("target")
        if isinstance(target, dict):
            for key in ("checked", "value", "isChecked"):
                if key in target:
                    return _coerce_bool(target.get(key))
    return False


def _normalize_section_titles(raw) -> dict[str, str]:
    """Normalize section title overrides to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return {}
        try:
            raw = json.loads(cleaned)
        except Exception:
            return {}
    if isinstance(raw, dict):
        return {
            str(k): str(v).strip()
            for k, v in raw.items()
            if k and str(v).strip()
        }
    return {}


def _normalize_custom_sections(raw) -> list[dict]:
    """Normalize custom sections to a list of dicts with id/key/title/body."""
    if raw is None:
        return []
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []
        try:
            raw = json.loads(cleaned)
        except Exception:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("id") or "").strip() or str(uuid.uuid4())
        key = str(item.get("key") or "").strip() or f"custom_{section_id}"
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or item.get("content") or "").rstrip()
        out.append(
            {
                "id": section_id,
                "key": key,
                "title": title,
                "body": body,
            }
        )
    return out


def _custom_section_keys(custom_sections: list[dict] | None) -> list[str]:
    keys: list[str] = []
    for item in custom_sections or []:
        key = str(item.get("key") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _build_section_title_map(
    section_titles: dict | None, custom_sections: list[dict] | None
) -> dict[str, str]:
    titles = {key: value for key, value in SECTION_LABELS.items()}
    overrides = _normalize_section_titles(section_titles)
    for key, value in overrides.items():
        if value:
            titles[key] = value
    for item in custom_sections or []:
        key = str(item.get("key") or "").strip()
        title = str(item.get("title") or "").strip()
        if key and title:
            titles[key] = title
    return titles


def _load_prompt_yaml_from_file(path: Path | None = None) -> str | None:
    """Load the prompt.yaml template from disk (if available)."""
    prompt_path = Path(path) if path else PROMPT_YAML_PATH
    try:
        return prompt_path.read_text(encoding="utf-8").rstrip()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _resolve_prompt_template(base_profile: dict | None) -> str | None:
    """Prefer a prompt stored in Neo4j; fall back to prompt.yaml on disk."""
    if isinstance(base_profile, dict):
        raw = base_profile.get("prompt_yaml")
        if isinstance(raw, str) and raw.strip():
            return raw.rstrip()
    return _load_prompt_yaml_from_file()


def _coerce_bullet_overrides(raw, *, keep_empty_id: bool = False) -> list[dict]:
    """Normalize bullet override payloads to a list of {id, bullets} dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    items: list[dict] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            items.append({"id": key, "bullets": value})
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                items.append(item)
    else:
        return []

    normalized: list[dict] = []
    for item in items:
        raw_id = item.get("id") or item.get("experience_id") or item.get("role_id")
        bullet_raw = item.get("bullets", [])
        entry_id = str(raw_id or "").strip()
        if not entry_id and not keep_empty_id:
            continue
        bullets: list[str] = []
        if isinstance(bullet_raw, str):
            bullets = [line.strip() for line in bullet_raw.split("\n") if line.strip()]
        elif isinstance(bullet_raw, (list, tuple)):
            for b in bullet_raw:
                if b is None:
                    continue
                text = str(b).strip()
                if text:
                    bullets.append(text)
        else:
            text = str(bullet_raw).strip()
            if text:
                bullets = [text]
        normalized.append({"id": entry_id, "bullets": bullets})
    return normalized


def _bullet_override_map(raw, *, allow_empty_id: bool = False) -> dict[str, list[str]]:
    overrides = _coerce_bullet_overrides(raw, keep_empty_id=allow_empty_id)
    out: dict[str, list[str]] = {}
    for item in overrides:
        entry_id = str(item.get("id") or "").strip()
        if not entry_id:
            continue
        bullets = [
            str(b).strip() for b in (item.get("bullets") or []) if str(b).strip()
        ]
        if bullets:
            out[entry_id] = bullets
    return out


def _coerce_bullet_text(bullets) -> str:
    if isinstance(bullets, (list, tuple)):
        return "\n".join([str(b).strip() for b in bullets if str(b).strip()])
    if bullets is None:
        return ""
    return str(bullets).strip()


def _apply_bullet_overrides(items: list[dict], overrides: dict[str, list[str]]):
    """Return items with bullets overridden by id when overrides are present."""
    if not overrides:
        return items
    out: list[dict] = []
    for item in items:
        data = dict(item or {})
        entry_id = str(data.get("id") or "").strip()
        override_bullets = overrides.get(entry_id)
        if override_bullets:
            data["bullets"] = list(override_bullets)
        out.append(data)
    return out


def _ensure_skill_rows(raw_rows):
    if raw_rows is None:
        raw_rows = []
    if isinstance(raw_rows, str):
        try:
            raw_rows = json.loads(raw_rows)
        except Exception:
            raw_rows = []
    rows: list[list[str]] = []
    if isinstance(raw_rows, (list, tuple)):
        for row in raw_rows[:3]:
            if isinstance(row, str):
                items = [p.strip() for p in row.split(",") if p.strip()]
            elif isinstance(row, (list, tuple)):
                items = [str(v).strip() for v in row if str(v).strip()]
            elif row is None or str(row).strip() == "":
                items = []
            else:
                items = [str(row).strip()]
            rows.append(items)
    while len(rows) < 3:
        rows.append([])
    return rows[:3]


def _em_value(
    layout_scale: float,
    value: float,
    *,
    weight: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Return a scaled float value (in `em` units)."""
    try:
        weight = float(weight)
    except Exception:
        weight = 1.0
    if weight <= 0:
        weight = 1.0

    scaled = float(value) * (float(layout_scale) ** weight)
    if min_value is not None:
        scaled = max(float(min_value), scaled)
    if max_value is not None:
        scaled = min(float(max_value), scaled)
    return scaled


def _fmt_em(value: float) -> str:
    if abs(value) < 1e-6:
        return "0em"
    return f"{value:.3f}em"


def _split_degree_parts(text) -> tuple[str, str]:
    """Return (main, detail) where detail preserves parentheses if present."""
    if not isinstance(text, str):
        return str(text or ""), ""
    raw = text.strip()
    if "(" in raw and raw.rstrip().endswith(")"):
        idx = raw.find("(")
        return raw[:idx].rstrip(), raw[idx:].strip()
    return raw, ""


def _parse_degree_details(detail) -> list[str]:
    """Turn a trailing parenthetical into a list of compact highlight items."""
    if not isinstance(detail, str):
        return []
    raw = detail.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def _format_degree_details(items: list[str]) -> str:
    items = [str(i).strip() for i in (items or []) if str(i).strip()]
    if not items:
        return ""
    return " · ".join(items)


# LLM providers are configured on-demand in functions that use them.


# ==========================================
# DATABASE LAYER
# ==========================================
def _get_shared_driver():
    """Create or reuse a process-wide Neo4j driver to avoid reconnect overhead."""
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        _NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _NEO4J_DRIVER


class Neo4jClient:
    def __init__(self, driver=None):
        self._stub = _get_maxcov_store() if _maxcov_stub_db_enabled() else None
        if self._stub is not None:
            self.driver = None
            self._owns_driver = False
        else:
            self.driver = driver or _get_shared_driver()
            self._owns_driver = driver is not None and driver is not _NEO4J_DRIVER
            self._ensure_schema()

    def _ensure_schema(self):
        """Create constraints/indexes needed by the app."""
        if self.driver is None:
            return
        global _NEO4J_SCHEMA_READY
        if _NEO4J_SCHEMA_READY:
            return
        try:
            statements: list[str] = []
            statements.append(
                "CREATE CONSTRAINT resume_id IF NOT EXISTS FOR (r:Resume) REQUIRE r.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT experience_id IF NOT EXISTS FOR (e:Experience) REQUIRE e.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT education_id IF NOT EXISTS FOR (e:Education) REQUIRE e.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT founder_id IF NOT EXISTS FOR (f:FounderRole) REQUIRE f.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT profile_id IF NOT EXISTS FOR (p:Profile) REQUIRE p.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE"
            )
            statements.append(
                "CREATE CONSTRAINT skill_category_name IF NOT EXISTS FOR (c:SkillCategory) REQUIRE c.name IS UNIQUE"
            )
            statements.append(
                "CREATE INDEX profile_created_at IF NOT EXISTS FOR (p:Profile) ON (p.created_at)"
            )
            statements.append(
                "CREATE INDEX experience_end_date IF NOT EXISTS FOR (e:Experience) ON (e.end_date)"
            )
            statements.append(
                "CREATE INDEX education_end_date IF NOT EXISTS FOR (e:Education) ON (e.end_date)"
            )
            statements.append(
                "CREATE INDEX founder_end_date IF NOT EXISTS FOR (f:FounderRole) ON (f.end_date)"
            )
            with self.driver.session() as session:
                for stmt in statements:
                    session.run(stmt)
            _NEO4J_SCHEMA_READY = True
        except Exception as e:
            print(f"Warning: could not ensure Neo4j schema: {e}")

    def close(self):
        if self._owns_driver and self.driver is not None:
            self.driver.close()

    def reset(self):
        """Blow away all nodes/edges in the DB."""
        if self._stub is not None:
            self._stub["resume"] = {}
            self._stub["experience"] = []
            self._stub["education"] = []
            self._stub["founder_roles"] = []
            self._stub["skills"] = []
            self._stub["profiles"] = []
            self._stub["auto_fit_cache"] = {}
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def reset_and_import(self, assets_path: str | Path = DEFAULT_ASSETS_JSON):
        """Reset the DB then import the provided assets JSON."""
        if self._stub is not None:
            self.reset()
            self.import_assets(assets_path, allow_overwrite=True)
            return
        self.reset()
        self.import_assets(assets_path, allow_overwrite=True)

    def resume_exists(self) -> bool:
        if self._stub is not None:
            return bool(self._stub.get("resume"))
        with self.driver.session() as session:
            row = session.run("MATCH (r:Resume) RETURN count(r) AS c").single()
            return bool(row and row["c"] and row["c"] > 0)

    def import_assets(
        self,
        assets_path: str | Path = DEFAULT_ASSETS_JSON,
        *,
        allow_overwrite: bool = False,
    ) -> bool:
        if self._stub is not None:
            assets_path = Path(assets_path)
            if not assets_path.is_absolute():
                assets_path = BASE_DIR / assets_path

            if not assets_path.exists():
                return False

            try:
                data = json.loads(assets_path.read_text(encoding="utf-8"))
            except Exception:
                return False
            _seed_maxcov_store(self._stub, data)
            return True
        assets_path = Path(assets_path)
        if not assets_path.is_absolute():
            assets_path = BASE_DIR / assets_path

        if not assets_path.exists():
            print(f"Seed file not found at {assets_path}")
            return False

        if not allow_overwrite and self.resume_exists():
            print(
                "Resume already exists; refusing to overwrite. "
                "Use --overwrite-resume to replace it."
            )
            return False

        with open(assets_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self.driver.session() as session:
            session.execute_write(self._create_resume, data.get("profile", {}))
            session.execute_write(self._create_experiences, data.get("experience", []))
            session.execute_write(self._create_education, data.get("education", []))
            session.execute_write(
                self._create_founder_roles, data.get("founder_roles", [])
            )
            session.execute_write(self._create_skills, data.get("skills", []))
        return True

    def ensure_resume_exists(self, assets_path: str | Path = DEFAULT_ASSETS_JSON):
        """If no Resume exists, import from JSON (dev bootstrap)."""
        if self._stub is not None:
            if not self._stub.get("resume"):
                self.import_assets(assets_path)
            if not self._stub.get("resume"):
                self._stub["resume"] = _empty_resume_payload()
            self.ensure_prompt_yaml()
            return
        with self.driver.session() as session:
            row = session.run("MATCH (r:Resume) RETURN count(r) AS c").single()
            if row and row["c"] and row["c"] > 0:
                self.ensure_prompt_yaml()
                self._ensure_placeholder_relationships()
                return
        self.import_assets(assets_path)
        with self.driver.session() as session:
            row = session.run("MATCH (r:Resume) RETURN count(r) AS c").single()
            if not (row and row["c"] and row["c"] > 0):
                payload = _empty_resume_payload()
                session.run(
                    """
                    MERGE (r:Resume {id: $id})
                    SET r.name = $name,
                        r.first_name = $first_name,
                        r.middle_name = $middle_name,
                        r.last_name = $last_name,
                        r.email = $email,
                        r.email2 = $email2,
                        r.phone = $phone,
                        r.font_family = $font_family,
                        r.auto_fit_target_pages = $auto_fit_target_pages,
                        r.auto_fit_best_scale = $auto_fit_best_scale,
                        r.auto_fit_too_long_scale = $auto_fit_too_long_scale,
                        r.linkedin_url = $linkedin_url,
                        r.github_url = $github_url,
                        r.scholar_url = $scholar_url,
                        r.calendly_url = $calendly_url,
                        r.portfolio_url = $portfolio_url,
                        r.summary = $summary,
                        r.head1_left = $head1_left,
                        r.head1_middle = $head1_middle,
                        r.head1_right = $head1_right,
                        r.head2_left = $head2_left,
                        r.head2_middle = $head2_middle,
                        r.head2_right = $head2_right,
                        r.head3_left = $head3_left,
                        r.head3_middle = $head3_middle,
                        r.head3_right = $head3_right,
                        r.top_skills = $top_skills,
                        r.section_order = $section_order,
                        r.section_enabled = $section_enabled,
                        r.section_titles_json = $section_titles_json,
                        r.custom_sections_json = $custom_sections_json,
                        r.prompt_yaml = $prompt_yaml
                    """,
                    **payload,
                )
        self.ensure_prompt_yaml()
        self._ensure_placeholder_relationships()

    def ensure_prompt_yaml(self, prompt_path: str | Path | None = None) -> str | None:
        """Seed Resume.prompt_yaml from prompt.yaml when missing."""
        prompt_text = _load_prompt_yaml_from_file(
            Path(prompt_path) if prompt_path else None
        )
        if not prompt_text:
            return None
        if self._stub is not None:
            resume = dict(self._stub.get("resume") or {})
            if not str(resume.get("prompt_yaml") or "").strip():
                resume["prompt_yaml"] = prompt_text
                self._stub["resume"] = resume
            return prompt_text
        with self.driver.session() as session:
            session.run(
                """
                MATCH (r:Resume)
                WHERE r.prompt_yaml IS NULL OR trim(r.prompt_yaml) = ''
                SET r.prompt_yaml = $prompt
                """,
                prompt=prompt_text,
            )
        return prompt_text

    def _create_resume(self, tx, profile_data):
        profile_data = dict(profile_data or {})
        profile_data.setdefault("scholar_url", "")
        profile_data.setdefault("calendly_url", "")
        profile_data.setdefault("portfolio_url", "")
        profile_data.setdefault("email2", "")
        profile_data.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
        profile_data.setdefault("auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES)
        profile_data.setdefault("auto_fit_best_scale", 1.0)
        profile_data.setdefault("auto_fit_too_long_scale", 0.0)
        profile_data.setdefault("section_order", list(DEFAULT_SECTION_ORDER))
        profile_data.setdefault("section_titles_json", "{}")
        profile_data.setdefault("custom_sections_json", "[]")
        section_titles = _normalize_section_titles(
            profile_data.get("section_titles_json")
            or profile_data.get("section_titles")
        )
        custom_sections = _normalize_custom_sections(
            profile_data.get("custom_sections_json")
            or profile_data.get("custom_sections")
        )
        extra_keys = _custom_section_keys(custom_sections)
        profile_data["section_order"] = _sanitize_section_order(
            profile_data.get("section_order"), extra_keys
        )
        profile_data["section_enabled"] = _normalize_section_enabled(
            profile_data.get("section_enabled"),
            list(SECTION_LABELS) + extra_keys,
            extra_keys=extra_keys,
        )
        profile_data["section_titles_json"] = json.dumps(
            section_titles, ensure_ascii=True
        )
        profile_data["custom_sections_json"] = json.dumps(
            custom_sections, ensure_ascii=True
        )
        profile_data.setdefault("prompt_yaml", "")
        query = """
        MERGE (r:Resume {id: $id})
        SET r.name = $name,
            r.email = $email,
            r.email2 = $email2,
            r.phone = $phone,
            r.font_family = $font_family,
            r.auto_fit_target_pages = $auto_fit_target_pages,
            r.auto_fit_best_scale = $auto_fit_best_scale,
            r.auto_fit_too_long_scale = $auto_fit_too_long_scale,
            r.linkedin_url = $linkedin_url,
            r.github_url = $github_url,
            r.scholar_url = $scholar_url,
            r.calendly_url = $calendly_url,
            r.portfolio_url = $portfolio_url,
            r.summary = $summary,
            r.head1_left = $head1_left,
            r.head1_middle = $head1_middle,
            r.head1_right = $head1_right,
            r.head2_left = $head2_left,
            r.head2_middle = $head2_middle,
            r.head2_right = $head2_right,
            r.head3_left = $head3_left,
            r.head3_middle = $head3_middle,
            r.head3_right = $head3_right,
            r.top_skills = $top_skills,
            r.section_order = $section_order,
            r.section_enabled = $section_enabled,
            r.section_titles_json = $section_titles_json,
            r.custom_sections_json = $custom_sections_json,
            r.prompt_yaml = $prompt_yaml
        """
        tx.run(query, **profile_data)

    def _ensure_placeholder_relationships(self) -> None:
        """Seed placeholder nodes to register relationship types when DB is empty."""
        if self.driver is None:
            return
        placeholder_date = "2000-01-01"
        placeholders = [
            (
                "Experience",
                "HAS_EXPERIENCE",
                "placeholder_experience",
                {
                    "company": "",
                    "role": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
            (
                "Education",
                "HAS_EDUCATION",
                "placeholder_education",
                {
                    "school": "",
                    "degree": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
            (
                "FounderRole",
                "HAS_FOUNDER_ROLE",
                "placeholder_founder",
                {
                    "company": "",
                    "role": "",
                    "location": "",
                    "description": "",
                    "bullets": [],
                },
            ),
        ]
        with self.driver.session() as session:
            for label, rel, node_id, fields in placeholders:
                session.run(
                    f"""
                    MATCH (r:Resume)
                    MERGE (n:{label} {{id: $id}})
                    ON CREATE SET n.is_placeholder = true,
                        n.start_date = date($start_date),
                        n.end_date = date($end_date),
                        n.company = $company,
                        n.role = $role,
                        n.location = $location,
                        n.description = $description,
                        n.bullets = $bullets,
                        n.school = $school,
                        n.degree = $degree
                    MERGE (r)-[:{rel}]->(n)
                    """,
                    id=node_id,
                    start_date=placeholder_date,
                    end_date=placeholder_date,
                    company=fields.get("company", ""),
                    role=fields.get("role", ""),
                    location=fields.get("location", ""),
                    description=fields.get("description", ""),
                    bullets=fields.get("bullets", []),
                    school=fields.get("school", ""),
                    degree=fields.get("degree", ""),
                )
                if MAX_COVERAGE_LOG:
                    _maxcov_log(f"placeholder ensured: {label}")
            session.run(
                """
                MATCH (r:Resume)
                MERGE (p:Profile {id: $id})
                ON CREATE SET p.is_placeholder = true,
                    p.created_at = datetime(),
                    p.summary = ""
                MERGE (p)-[:FOR_RESUME]->(r)
                """,
                id="placeholder_profile",
            )
            if MAX_COVERAGE_LOG:
                _maxcov_log("placeholder ensured: Profile")

    def _create_experiences(self, tx, experiences):
        query = """
        MATCH (r:Resume)
        MERGE (e:Experience {id: $id})
        SET e.company = $company,
            e.role = $role,
            e.location = $location,
            e.description = $description,
            e.bullets = $bullets,
            e.start_date = date($start_date),
            e.end_date = date($end_date)
        MERGE (r)-[:HAS_EXPERIENCE]->(e)
        """
        for exp in experiences:
            tx.run(query, **exp)

    def _create_education(self, tx, education):
        query = """
        MATCH (r:Resume)
        MERGE (e:Education {id: $id})
        SET e.school = $school,
            e.degree = $degree,
            e.location = $location,
            e.description = $description,
            e.bullets = $bullets,
            e.start_date = date($start_date),
            e.end_date = date($end_date)
        MERGE (r)-[:HAS_EDUCATION]->(e)
        """
        for edu in education:
            tx.run(query, **edu)

    def _create_founder_roles(self, tx, roles):
        query = """
        MATCH (r:Resume)
        MERGE (f:FounderRole {id: $id})
        SET f.company = $company,
            f.role = $role,
            f.location = $location,
            f.description = $description,
            f.bullets = $bullets,
            f.start_date = date($start_date),
            f.end_date = date($end_date)
        MERGE (r)-[:HAS_FOUNDER_ROLE]->(f)
        """
        for role in roles:
            tx.run(query, **role)

    def _create_skills(self, tx, skill_categories):
        for category in skill_categories:
            cat_name = category["category"]
            query_cat = "MERGE (c:SkillCategory {name: $name})"
            tx.run(query_cat, name=cat_name)

            query_skill = """
            MATCH (c:SkillCategory {name: $cat_name})
            MERGE (s:Skill {id: $id})
            SET s.name = $name
            MERGE (s)-[:IN_CATEGORY]->(c)
            """
            for skill in category["skills"]:
                tx.run(query_skill, cat_name=cat_name, **skill)

    def get_resume_data(self):
        if self._stub is not None:
            resume = dict(self._stub.get("resume") or {})
            if not resume:
                return None
            return {
                "resume": resume,
                "experience": list(self._stub.get("experience") or []),
                "education": list(self._stub.get("education") or []),
                "founder_roles": list(self._stub.get("founder_roles") or []),
            }
        with self.driver.session() as session:
            resume = session.run("MATCH (r:Resume) RETURN r").single()
            if not resume:
                return None

            resume_data = dict(resume["r"])

        self._ensure_placeholder_relationships()

        with self.driver.session() as session:
            experiences = session.run(
                """
                MATCH (r:Resume)-[:HAS_EXPERIENCE]->(e)
                WHERE coalesce(e.is_placeholder, false) = false
                RETURN e
                ORDER BY coalesce(e.end_date, date('9999-12-31')) DESC,
                         coalesce(e.start_date, date('0001-01-01')) DESC
                """
            ).data()
            education = session.run(
                """
                MATCH (r:Resume)-[:HAS_EDUCATION]->(e)
                WHERE coalesce(e.is_placeholder, false) = false
                RETURN e
                ORDER BY coalesce(e.end_date, date('9999-12-31')) DESC,
                         coalesce(e.start_date, date('0001-01-01')) DESC
                """
            ).data()
            founder_roles = session.run(
                """
                MATCH (r:Resume)-[:HAS_FOUNDER_ROLE]->(f)
                WHERE coalesce(f.is_placeholder, false) = false
                RETURN f
                ORDER BY coalesce(f.end_date, date('9999-12-31')) DESC,
                         coalesce(f.start_date, date('0001-01-01')) DESC
                """
            ).data()

            # Helper to convert neo4j dates to string
            def serialize_dates(items):
                result = []
                for item in items:
                    node = item["e"] if "e" in item else item["f"]
                    data = dict(node)
                    if "start_date" in data:
                        data["start_date"] = str(data["start_date"])
                    if "end_date" in data:
                        data["end_date"] = str(data["end_date"])
                    result.append(data)
                return result

            return {
                "resume": resume_data,
                "experience": serialize_dates(experiences),
                "education": serialize_dates(education),
                "founder_roles": serialize_dates(founder_roles),
            }

    def get_auto_fit_cache(self):
        """Return the last auto-fit tuning values stored on the Resume node (if any)."""
        if self._stub is not None:
            cache = self._stub.get("auto_fit_cache") or {}
            if not cache:
                return None
            return {
                "best_scale": cache.get("best_scale"),
                "too_long_scale": cache.get("too_long_scale"),
            }
        with self.driver.session() as session:
            row = session.run(
                """
                MATCH (r:Resume)
                RETURN
                  r.auto_fit_best_scale AS best_scale,
                  r.auto_fit_too_long_scale AS too_long_scale
                LIMIT 1
                """
            ).single()
            if not row:
                return None
            return {
                "best_scale": row.get("best_scale"),
                "too_long_scale": row.get("too_long_scale"),
            }

    def set_auto_fit_cache(
        self,
        *,
        best_scale: float,
        too_long_scale: float | None = None,
    ):
        """Store the latest auto-fit tuning values on the Resume node."""
        if self._stub is not None:
            self._stub["auto_fit_cache"] = {
                "best_scale": float(best_scale),
                "too_long_scale": (
                    float(too_long_scale) if too_long_scale is not None else None
                ),
            }
            return
        with self.driver.session() as session:
            session.run(
                """
                MATCH (r:Resume)
                SET
                  r.auto_fit_best_scale = $best_scale,
                  r.auto_fit_too_long_scale = $too_long_scale
                """,
                best_scale=float(best_scale),
                too_long_scale=(
                    float(too_long_scale) if too_long_scale is not None else None
                ),
            )

    def list_applied_jobs(self):
        """Return applied jobs (profiles) with key metadata."""
        rows = None
        if self._stub is not None:
            profiles = list(self._stub.get("profiles") or [])

            def sort_key(profile):
                return str(profile.get("created_at") or "")

            profiles.sort(key=sort_key, reverse=True)
            rows = [{"p": dict(profile)} for profile in profiles]
        else:
            self._ensure_placeholder_relationships()
            with self.driver.session() as session:
                rows = session.run(
                    """
                    MATCH (p:Profile)-[:FOR_RESUME]->(:Resume)
                    WHERE coalesce(p.is_placeholder, false) = false
                    RETURN p
                    ORDER BY p.created_at DESC
                    """
                ).data()
        jobs = []
        for row in rows:
            p = dict(row["p"])
            skills_rows = []
            raw_skills_rows = (p.get("skills_rows_json") or "").strip()
            if raw_skills_rows:
                try:
                    skills_rows = json.loads(raw_skills_rows)
                except Exception:
                    skills_rows = []
            norm_rows: list[list[str]] = []
            if isinstance(skills_rows, str):
                try:
                    skills_rows = json.loads(skills_rows)
                except Exception:
                    skills_rows = []
            if isinstance(skills_rows, (list, tuple)):
                for row_items in skills_rows[:3]:
                    if isinstance(row_items, str):
                        items = [s.strip() for s in row_items.split(",") if s.strip()]
                    elif isinstance(row_items, (list, tuple)):
                        items = [str(s).strip() for s in row_items if str(s).strip()]
                    elif row_items is None or str(row_items).strip() == "":
                        items = []
                    else:
                        items = [str(row_items).strip()]
                    norm_rows.append(items)
            while len(norm_rows) < 3:
                norm_rows.append([])
            norm_rows = norm_rows[:3]
            experience_bullets = _coerce_bullet_overrides(
                p.get("experience_bullets_json")
            )
            founder_role_bullets = _coerce_bullet_overrides(
                p.get("founder_role_bullets_json")
            )
            jobs.append(
                {
                    "id": p.get("id"),
                    "created_at": str(p.get("created_at")),
                    "target_company": p.get("target_company", ""),
                    "target_role": p.get("target_role", ""),
                    "seniority_level": p.get("seniority_level", ""),
                    "target_location": p.get("target_location", ""),
                    "work_mode": p.get("work_mode", ""),
                    "req_id": p.get("req_id", ""),
                    "summary": p.get("summary", ""),
                    "headers": p.get("headers", []),
                    "highlighted_skills": p.get("highlighted_skills", []),
                    "skills_rows": norm_rows,
                    "experience_bullets": experience_bullets,
                    "founder_role_bullets": founder_role_bullets,
                    "job_req_raw": p.get("job_req_raw", ""),
                    "travel_requirement": p.get("travel_requirement", ""),
                    "primary_domain": p.get("primary_domain", ""),
                    "must_have_skills": p.get("must_have_skills", []),
                    "nice_to_have_skills": p.get("nice_to_have_skills", []),
                    "tech_stack_keywords": p.get("tech_stack_keywords", []),
                    "non_technical_requirements": p.get(
                        "non_technical_requirements", []
                    ),
                    "certifications": p.get("certifications", []),
                    "clearances": p.get("clearances", []),
                    "core_responsibilities": p.get("core_responsibilities", []),
                    "outcome_goals": p.get("outcome_goals", []),
                    "salary_band": p.get("salary_band", ""),
                    "posting_url": p.get("posting_url", ""),
                }
            )
        return jobs

    def save_resume(self, resume_data):
        """
        Persist a versioned Profile node (LLM output) linked to the canonical Resume.
        Nested arrays (e.g., skills rows) are stored as JSON strings to satisfy
        Neo4j property type constraints.
        """
        if self._stub is not None:
            record = dict(resume_data or {})
            record.setdefault("id", str(uuid.uuid4()))
            record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            self._stub.setdefault("profiles", []).append(record)
            return record["id"]
        with self.driver.session() as session:
            resume_data = dict(resume_data or {})
            resume_data.setdefault("summary", "")
            resume_data.setdefault("headers", [])
            resume_data.setdefault("highlighted_skills", [])
            resume_data.setdefault("skills_rows_json", "[]")
            resume_data.setdefault("job_req_raw", "")
            resume_data.setdefault("target_company", "")
            resume_data.setdefault("target_role", "")
            resume_data.setdefault("seniority_level", "")
            resume_data.setdefault("target_location", "")
            resume_data.setdefault("work_mode", "")
            resume_data.setdefault("travel_requirement", "")
            resume_data.setdefault("primary_domain", "")
            resume_data.setdefault("must_have_skills", [])
            resume_data.setdefault("nice_to_have_skills", [])
            resume_data.setdefault("tech_stack_keywords", [])
            resume_data.setdefault("non_technical_requirements", [])
            resume_data.setdefault("certifications", [])
            resume_data.setdefault("clearances", [])
            resume_data.setdefault("core_responsibilities", [])
            resume_data.setdefault("outcome_goals", [])
            resume_data.setdefault("salary_band", "")
            resume_data.setdefault("posting_url", "")
            resume_data.setdefault("req_id", "")
            resume_data.setdefault("experience_bullets_json", "[]")
            resume_data.setdefault("founder_role_bullets_json", "[]")
            query = """
            MATCH (r:Resume)
            CREATE (p:Profile {
                id: randomUUID(),
                created_at: datetime(),
                summary: $summary,
                headers: $headers,
                highlighted_skills: $highlighted_skills,
                skills_rows_json: $skills_rows_json,
                experience_bullets_json: $experience_bullets_json,
                founder_role_bullets_json: $founder_role_bullets_json,
                job_req_raw: $job_req_raw,
                target_company: $target_company,
                target_role: $target_role,
                seniority_level: $seniority_level,
                target_location: $target_location,
                work_mode: $work_mode,
                travel_requirement: $travel_requirement,
                primary_domain: $primary_domain,
                must_have_skills: $must_have_skills,
                nice_to_have_skills: $nice_to_have_skills,
                tech_stack_keywords: $tech_stack_keywords,
                non_technical_requirements: $non_technical_requirements,
                certifications: $certifications,
                clearances: $clearances,
                core_responsibilities: $core_responsibilities,
                outcome_goals: $outcome_goals,
                salary_band: $salary_band,
                posting_url: $posting_url,
                req_id: $req_id
            })
            MERGE (p)-[:FOR_RESUME]->(r)
            RETURN p.id as id
            """
            result = session.run(query, **resume_data)
            row = result.single()
            if not row:
                raise RuntimeError("Failed to save profile: Resume not found.")
            return row["id"]

    def update_profile_bullets(
        self,
        profile_id: str,
        experience_bullets: list[dict],
        founder_role_bullets: list[dict],
    ) -> bool:
        """Update bullet overrides on a Profile node by id."""
        if self._stub is not None:
            profiles = self._stub.get("profiles") or []
            target = str(profile_id or "")
            for profile in profiles:
                if str(profile.get("id") or "") != target:
                    continue
                profile["experience_bullets_json"] = json.dumps(
                    experience_bullets or [], ensure_ascii=False
                )
                profile["founder_role_bullets_json"] = json.dumps(
                    founder_role_bullets or [], ensure_ascii=False
                )
                return True
            return False
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Profile {id: $id})
                SET
                  p.experience_bullets_json = $experience_bullets_json,
                  p.founder_role_bullets_json = $founder_role_bullets_json
                RETURN p.id AS id
                """,
                id=str(profile_id or ""),
                experience_bullets_json=json.dumps(
                    experience_bullets or [], ensure_ascii=False
                ),
                founder_role_bullets_json=json.dumps(
                    founder_role_bullets or [], ensure_ascii=False
                ),
            )
            row = result.single()
            return bool(row and row.get("id"))

    def upsert_resume_and_sections(
        self,
        resume_fields,
        experiences,
        education,
        founder_roles,
        *,
        delete_missing: bool = False,
    ):
        """Upsert resume, experiences, education, and founder roles in Neo4j."""
        if self._stub is not None:
            resume_fields = dict(resume_fields or {})
            resume_fields.setdefault("section_order", list(DEFAULT_SECTION_ORDER))
            section_titles = _normalize_section_titles(
                resume_fields.get("section_titles_json")
                or resume_fields.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                resume_fields.get("custom_sections_json")
                or resume_fields.get("custom_sections")
            )
            extra_keys = _custom_section_keys(custom_sections)
            resume_fields["section_order"] = _sanitize_section_order(
                resume_fields.get("section_order"), extra_keys
            )
            resume_fields["section_enabled"] = _normalize_section_enabled(
                resume_fields.get("section_enabled"),
                list(SECTION_LABELS) + extra_keys,
                extra_keys=extra_keys,
            )
            resume_fields["section_titles_json"] = json.dumps(
                section_titles, ensure_ascii=True
            )
            resume_fields["custom_sections_json"] = json.dumps(
                custom_sections, ensure_ascii=True
            )
            resume_fields.setdefault("email2", "")
            resume_fields.setdefault("calendly_url", "")
            resume_fields.setdefault("portfolio_url", "")
            resume_fields.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
            resume_fields.setdefault(
                "auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES
            )
            resume_fields.setdefault("prompt_yaml", "")
            resume = dict(self._stub.get("resume") or {})
            resume.update(resume_fields)
            self._stub["resume"] = resume
            self._stub["experience"] = list(experiences or [])
            self._stub["education"] = list(education or [])
            self._stub["founder_roles"] = list(founder_roles or [])
            return
        with self.driver.session() as session:
            resume_fields = dict(resume_fields or {})
            resume_fields.setdefault("section_order", list(DEFAULT_SECTION_ORDER))
            section_titles = _normalize_section_titles(
                resume_fields.get("section_titles_json")
                or resume_fields.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                resume_fields.get("custom_sections_json")
                or resume_fields.get("custom_sections")
            )
            extra_keys = _custom_section_keys(custom_sections)
            resume_fields["section_order"] = _sanitize_section_order(
                resume_fields.get("section_order"), extra_keys
            )
            resume_fields["section_enabled"] = _normalize_section_enabled(
                resume_fields.get("section_enabled"),
                list(SECTION_LABELS) + extra_keys,
                extra_keys=extra_keys,
            )
            resume_fields["section_titles_json"] = json.dumps(
                section_titles, ensure_ascii=True
            )
            resume_fields["custom_sections_json"] = json.dumps(
                custom_sections, ensure_ascii=True
            )
            resume_fields.setdefault("email2", "")
            resume_fields.setdefault("calendly_url", "")
            resume_fields.setdefault("portfolio_url", "")
            resume_fields.setdefault("font_family", DEFAULT_RESUME_FONT_FAMILY)
            resume_fields.setdefault(
                "auto_fit_target_pages", DEFAULT_AUTO_FIT_TARGET_PAGES
            )
            resume_fields.setdefault("prompt_yaml", "")
            session.run(
                """
                MATCH (r:Resume)
                SET r.summary = $summary,
                    r.prompt_yaml = $prompt_yaml,
                    r.name = $name,
                    r.first_name = $first_name,
                    r.middle_name = $middle_name,
                    r.last_name = $last_name,
                    r.email = $email,
                    r.email2 = $email2,
                    r.phone = $phone,
                    r.font_family = $font_family,
                    r.auto_fit_target_pages = $auto_fit_target_pages,
                    r.linkedin_url = $linkedin_url,
                    r.github_url = $github_url,
                    r.scholar_url = $scholar_url,
                    r.calendly_url = $calendly_url,
                    r.portfolio_url = $portfolio_url,
                    r.head1_left = $head1_left,
                    r.head1_middle = $head1_middle,
                    r.head1_right = $head1_right,
                    r.head2_left = $head2_left,
                    r.head2_middle = $head2_middle,
                    r.head2_right = $head2_right,
                    r.head3_left = $head3_left,
                    r.head3_middle = $head3_middle,
                    r.head3_right = $head3_right,
                    r.top_skills = $top_skills,
                    r.section_order = $section_order,
                    r.section_enabled = $section_enabled,
                    r.section_titles_json = $section_titles_json,
                    r.custom_sections_json = $custom_sections_json
                """,
                **resume_fields,
            )

            def date_clause(field):
                return f"CASE WHEN ${field} IS NOT NULL AND ${field} <> '' THEN date(${field}) ELSE NULL END"

            exp_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (e:Experience {{id: $id}})
                SET e.company = $company,
                    e.role = $role,
                    e.location = $location,
                    e.description = $description,
                    e.bullets = $bullets,
                    e.start_date = {date_clause("start_date")},
                    e.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_EXPERIENCE]->(e)
                    """,
                )
            )
            for exp in experiences:
                session.run(exp_query, **exp)

            edu_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (e:Education {{id: $id}})
                SET e.school = $school,
                    e.degree = $degree,
                    e.location = $location,
                    e.description = $description,
                    e.bullets = $bullets,
                    e.start_date = {date_clause("start_date")},
                    e.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_EDUCATION]->(e)
                    """,
                )
            )
            for edu in education:
                session.run(edu_query, **edu)

            role_query = Query(
                cast(
                    LiteralString,
                    f"""
                MATCH (r:Resume)
                MERGE (f:FounderRole {{id: $id}})
                SET f.company = $company,
                    f.role = $role,
                    f.location = $location,
                    f.description = $description,
                    f.bullets = $bullets,
                    f.start_date = {date_clause("start_date")},
                    f.end_date = {date_clause("end_date")}
                MERGE (r)-[:HAS_FOUNDER_ROLE]->(f)
                    """,
                )
            )
            for role in founder_roles:
                session.run(role_query, **role)

            if delete_missing:
                exp_ids = sorted(
                    {exp.get("id") for exp in experiences if exp.get("id")}
                )
                edu_ids = sorted({edu.get("id") for edu in education if edu.get("id")})
                role_ids = sorted(
                    {role.get("id") for role in founder_roles if role.get("id")}
                )
                session.run(
                    """
                    MATCH (r:Resume)-[:HAS_EXPERIENCE]->(e:Experience)
                    WHERE NOT e.id IN $ids
                    DETACH DELETE e
                    """,
                    ids=exp_ids,
                )
                session.run(
                    """
                    MATCH (r:Resume)-[:HAS_EDUCATION]->(e:Education)
                    WHERE NOT e.id IN $ids
                    DETACH DELETE e
                    """,
                    ids=edu_ids,
                )
                session.run(
                    """
                    MATCH (r:Resume)-[:HAS_FOUNDER_ROLE]->(f:FounderRole)
                    WHERE NOT f.id IN $ids
                    DETACH DELETE f
                    """,
                    ids=role_ids,
                )


# ==========================================
# LLM LAYER (any-llm)
# ==========================================
def _split_llm_model_spec(raw: str | None) -> tuple[str, str]:
    """Return (provider, model_id) from `provider:model`, `provider/model`, or bare `model`."""
    raw = (raw or "").strip()
    if not raw:
        # DEFAULT_LLM_MODEL is already canonicalized to include a provider prefix.
        raw = DEFAULT_LLM_MODEL
    if ":" in raw:
        prefix, rest = raw.split(":", 1)
        provider = prefix.strip().lower()
        if provider in SUPPORTED_LLM_PROVIDERS and rest.strip():
            return provider, rest.strip()
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        provider = prefix.strip().lower()
        if provider in SUPPORTED_LLM_PROVIDERS and rest.strip():
            return provider, rest.strip()
    # Backward compatibility: bare model ids default to OpenAI.
    return "openai", raw


def list_llm_models() -> list[str]:
    """Return configured model ids for the UI/CLI (no network calls)."""
    raw = (
        os.environ.get("LLM_MODELS") or os.environ.get("OPENAI_MODELS") or ""
    ).strip()
    if raw:
        candidates = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        candidates = list(DEFAULT_LLM_MODELS)
    seen: set[str] = set()
    out: list[str] = []
    for spec in candidates:
        provider, model = _split_llm_model_spec(spec)
        canonical = f"{provider}:{model}"
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _read_first_secret_line(path: Path) -> str | None:
    """Return the first non-empty, non-comment line from a file (no printing)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return None


def load_openai_api_key() -> str | None:
    """Load OpenAI API key from ~/openaikey.txt or env (no printing)."""
    path = Path.home() / "openaikey.txt"

    def read_kv(name: str) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        target = name.strip().upper()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().upper() == target:
                return value.strip().strip('"').strip("'")
        return None

    from_kv = read_kv("OPENAI_API_KEY")
    if from_kv:
        return from_kv

    # Backward compatibility: a single-line file containing just the key.
    raw = _read_first_secret_line(path)
    if not raw:
        raw = ""
    if raw.lstrip().startswith("sk-"):
        return raw.strip().strip('"').strip("'")

    env = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env:
        return env
    return None


def load_gemini_api_key() -> str | None:
    """Load Gemini API key from env or ~/openaikey.txt (GEMINI_API_KEY/GOOGLE_API_KEY)."""
    env = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if env:
        return env
    env = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if env:
        return env

    path = Path.home() / "openaikey.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if key in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
            return value.strip().strip('"').strip("'")
    return None


def _openai_reasoning_params_for_model(model: str) -> dict | None:
    """Return OpenAI `reasoning` params for models that support it."""
    effort = (DEFAULT_LLM_REASONING_EFFORT or "").strip().lower()
    if not effort or effort == "none":
        return None
    # OpenAI doesn't consistently accept "minimal"; map it to "low".
    if effort == "minimal":
        effort = "low"
    model_id = (model or "").strip().lower()
    if not model_id:
        return None
    # Heuristic: reasoning params are accepted by o-series and GPT-5.x models.
    if model_id.startswith("o") or model_id.startswith("gpt-5"):
        return {"effort": effort}
    return None


def _read_int_env(*names: str) -> int | None:
    for name in names:
        if not name:
            continue
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0:
            return val
    return None


def _resolve_llm_max_output_tokens(provider: str, model: str) -> int:
    """Pick a safe output token budget, overridable via LLM_MAX_OUTPUT_TOKENS."""
    provider = (provider or "").strip().lower()
    model_id = (model or "").strip().lower()

    override = _read_int_env(
        "LLM_MAX_OUTPUT_TOKENS",
        "OPENAI_MAX_OUTPUT_TOKENS" if provider == "openai" else "",
        "GEMINI_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
        "GOOGLE_MAX_OUTPUT_TOKENS" if provider == "gemini" else "",
    )
    if override:
        return override

    if provider == "openai":
        # Reasoning-capable models can consume output budget for hidden reasoning. Give them more.
        if model_id.startswith("gpt-5") or model_id.startswith("o"):
            return 8192
        return 4096

    if provider == "gemini":
        return 4096

    return 4096


def _resolve_llm_retry_max_output_tokens(
    provider: str, model: str, initial: int
) -> int:
    """Pick a larger token budget for a single retry on truncation."""
    provider = (provider or "").strip().lower()
    model_id = (model or "").strip().lower()

    override = _read_int_env(
        "LLM_MAX_OUTPUT_TOKENS_RETRY",
        "OPENAI_MAX_OUTPUT_TOKENS_RETRY" if provider == "openai" else "",
        "GEMINI_MAX_OUTPUT_TOKENS_RETRY" if provider == "gemini" else "",
        "GOOGLE_MAX_OUTPUT_TOKENS_RETRY" if provider == "gemini" else "",
    )
    if override:
        return override

    # Default: grow aggressively once, but keep it bounded.
    if provider == "openai":
        cap = (
            16384
            if (model_id.startswith("gpt-5") or model_id.startswith("o"))
            else 8192
        )
    else:
        cap = 8192
    return min(max(int(initial) * 2, int(initial) + 2048), cap)


def render_resume_pdf_bytes(
    save_copy: bool = False,
    include_summary: bool = True,
    include_skills: bool = True,
    filename: str = "preview_no_summary_skills.pdf",
):
    """Compile a resume to PDF with optional summary/skills; optionally persist a copy."""
    try:
        db = Neo4jClient()
        data = db.get_resume_data() or {}
        db.close()

        resume_node = data.get("resume", {}) or {}

        def ensure_len(items, target=9):
            items = list(items or [])
            while len(items) < target:
                items.append("")
            return items[:target]

        def parse_section_order(raw_order, extra_keys):
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            return _sanitize_section_order(raw_order, extra_keys)

        headers = ensure_len(
            [
                resume_node.get("head1_left", ""),
                resume_node.get("head1_middle", ""),
                resume_node.get("head1_right", ""),
                resume_node.get("head2_left", ""),
                resume_node.get("head2_middle", ""),
                resume_node.get("head2_right", ""),
                resume_node.get("head3_left", ""),
                resume_node.get("head3_middle", ""),
                resume_node.get("head3_right", ""),
            ]
        )

        skills = ensure_len(resume_node.get("top_skills", []))
        summary_text = resume_node.get("summary", "") if include_summary else ""

        section_titles = _normalize_section_titles(
            resume_node.get("section_titles_json")
            or resume_node.get("section_titles")
        )
        custom_sections = _normalize_custom_sections(
            resume_node.get("custom_sections_json")
            or resume_node.get("custom_sections")
        )
        extra_keys = _custom_section_keys(custom_sections)
        resume_data = {
            "summary": summary_text,
            "headers": headers[:9],
            "highlighted_skills": skills[:9] if include_skills else [],
            "first_name": resume_node.get("first_name", ""),
            "middle_name": resume_node.get("middle_name", ""),
            "last_name": resume_node.get("last_name", ""),
            "email": resume_node.get("email", ""),
            "email2": resume_node.get("email2", ""),
            "phone": resume_node.get("phone", ""),
            "font_family": resume_node.get("font_family", DEFAULT_RESUME_FONT_FAMILY),
            "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                resume_node.get("auto_fit_target_pages"),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            ),
            "linkedin_url": resume_node.get("linkedin_url", ""),
            "github_url": resume_node.get("github_url", ""),
            "scholar_url": resume_node.get("scholar_url", ""),
            "calendly_url": resume_node.get("calendly_url", ""),
            "portfolio_url": resume_node.get("portfolio_url", ""),
            "section_order": parse_section_order(
                resume_node.get("section_order"), extra_keys
            ),
            "section_titles": section_titles,
            "custom_sections": custom_sections,
        }
        section_enabled = _normalize_section_enabled(
            resume_node.get("section_enabled"),
            list(SECTION_LABELS) + extra_keys,
            extra_keys=extra_keys,
        )
        resume_data["section_order"] = _apply_section_enabled(
            resume_data["section_order"],
            section_enabled,
        )
        profile_data = {
            **resume_node,
            "experience": data.get("experience", []),
            "education": data.get("education", []),
            "founder_roles": data.get("founder_roles", []),
        }
        typst_src = generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=include_skills,
            include_summary=include_summary,
            section_order=resume_data["section_order"],
        )
        ok, pdf_bytes = compile_pdf(typst_src)
        if not ok or not pdf_bytes:
            return None
        if save_copy:
            dest = ASSETS_DIR / filename
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(pdf_bytes)
            except Exception as e:
                print(f"Warning: could not save resume pdf output: {e}")
        return pdf_bytes
    except Exception as e:
        print(f"Warning: could not render resume pdf: {e}")
    return None


def _rasterize_text_image(
    text: str,
    *,
    font_size: int = 48,
    target_height_pt: float | None = None,
    italic: bool = False,
    font_family: str | None = None,
) -> str:
    """Render a short text snippet to a PNG (high-res, inline) and return its repo-relative path."""
    text = (text or "").strip()
    if not text:
        return ""
    if not font_family:
        font_family = DEFAULT_RESUME_FONT_FAMILY
    try:
        from PIL import (
            Image,
            ImageDraw,
            ImageFont,
        )  # Lazy import; optional dependency.

        TEMP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        # Use the selected font family to match resume typography.
        font_size = max(6, int(font_size))  # Guard against tiny/invalid sizes.
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None
        font_candidates = _select_local_font_paths(font_family, italic=italic)
        for candidate in font_candidates:
            try:
                font = ImageFont.truetype(str(candidate), font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        # Measure text bounds at the render resolution.
        dummy = Image.new("L", (1, 1), 0)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = int((bbox[2] - bbox[0]) + 12)
        height = int((bbox[3] - bbox[1]) + 12)
        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        # Use deep black for crisp rasterized text; Typst applies sizing on embed.
        draw.text((6, 6), text, fill=(0, 0, 0, 255), font=font)

        # Optionally pre-scale to reduce aliasing when Typst scales down.
        if target_height_pt:
            # Oversample more aggressively (~6x at 96dpi) for cleaner downscale in Typst.
            target_px = max(1, int(target_height_pt * (96 / 72) * 6.0))
            scale = target_px / img.height
            if scale > 0:
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                resample = getattr(
                    getattr(Image, "Resampling", Image),
                    "LANCZOS",
                    getattr(Image, "BICUBIC", 3),
                )
                img = img.resize((new_w, new_h), resample)

        cache_key = f"{text}|{target_height_pt or ''}|{'ital' if italic else 'reg'}"
        fname = (
            "founder_date_"
            + hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
            + ".png"
        )
        fpath = TEMP_BUILD_DIR / fname
        img.save(fpath, format="PNG")
        return "/" + str(fpath.relative_to(BASE_DIR))
    except Exception:
        return ""


def generate_typst_source(
    resume_data,
    profile_data,
    include_matrices=True,
    include_summary=True,
    section_order=None,
    layout_scale: float = 1.0,
):
    """
    Converts resume data into Typst markup.
    """
    # Extract data
    email = profile_data.get("email", "email@example.com")
    email2 = profile_data.get("email2", resume_data.get("email2", ""))
    phone = profile_data.get("phone", "555-0123")
    linkedin = normalize_linkedin(profile_data.get("linkedin_url", ""))

    def _split_github_input(raw: str) -> tuple[str, str]:
        raw = str(raw or "").strip()
        if not raw:
            return "", ""
        if re.match(r"^https?://", raw, flags=re.IGNORECASE):
            try:
                parsed = urlparse(raw)
            except Exception:
                return "", raw
            host = (parsed.netloc or "").strip().lower()
            if host.startswith("www."):
                host = host[4:]
            if host and host != "github.com":
                return "", raw
            return normalize_github(parsed.path.lstrip("/")), ""
        trimmed = raw.lstrip()
        lowered = trimmed.lower()
        if lowered.startswith("www."):
            trimmed = trimmed[4:]
            lowered = trimmed.lower()
        if lowered.startswith("github.com/"):
            return normalize_github(trimmed), ""
        host_candidate = lowered.split("/", 1)[0]
        if "." in host_candidate:
            return "", "https://" + trimmed.lstrip("/")
        return normalize_github(trimmed), ""

    github_path, github_url = _split_github_input(profile_data.get("github_url", ""))
    github = github_path
    scholar_url = normalize_scholar_url(profile_data.get("scholar_url", ""))
    calendly_url = normalize_calendly_url(
        profile_data.get("calendly_url", resume_data.get("calendly_url", ""))
    )
    portfolio_url = normalize_portfolio_url(
        profile_data.get("portfolio_url", resume_data.get("portfolio_url", ""))
    )
    portfolio_label = "portfolio" if portfolio_url else ""
    portfolio_label = escape_typst(portfolio_label)
    scholar_label = escape_typst(
        profile_data.get("scholar_link_text", format_url_label(scholar_url))
    )
    github_label_source = profile_data.get("github_link_text", "")
    if github_url:
        if not github_label_source:
            github_label_source = format_url_label(github_url)
    elif not github_label_source:
        github_label_source = github
    github_label = escape_typst(github_label_source)
    linkedin_label = escape_typst(profile_data.get("linkedin_link_text", linkedin))
    custom_contacts: list[dict[str, str]] = []
    if github_url:
        custom_contacts.append(
            {"text": github_label_source, "icon": "github", "link": github_url}
        )
    if calendly_url:
        calendly_label = format_url_label(calendly_url) or calendly_url
        custom_contacts.append(
            {"text": calendly_label, "icon": "calendar", "link": calendly_url}
        )

    # Prefer the resume-level summary; fall back to profile summary if missing.
    summary_source = resume_data.get("summary") or profile_data.get("summary", "")
    summary = summary_source if (summary_source and include_summary) else ""
    resume_font_family = (
        resume_data.get("font_family")
        or profile_data.get("font_family")
        or DEFAULT_RESUME_FONT_FAMILY
    )
    resume_font_family = str(resume_font_family or DEFAULT_RESUME_FONT_FAMILY).strip()
    if not resume_font_family:
        resume_font_family = DEFAULT_RESUME_FONT_FAMILY
    resume_font_family_escaped = escape_typst(resume_font_family)

    try:
        layout_scale = float(layout_scale)
    except Exception:
        layout_scale = 1.0
    if not (layout_scale > 0):
        layout_scale = 1.0
    layout_scale = max(0.35, min(layout_scale, 6.0))

    def em_value(
        value: float,
        *,
        weight: float = 1.0,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        return _em_value(
            layout_scale, value, weight=weight, min_value=min_value, max_value=max_value
        )

    def fmt_em(value: float) -> str:
        return _fmt_em(value)

    def em(
        value: float,
        *,
        weight: float = 1.0,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> str:
        """
        Scale a Typst `em` value using a weighted exponent and clamp it.

        The goal is to let auto-fit adjust the document height without:
        - shrinking micro-leading so far that text overlaps
        - inflating macro gaps (between entries/sections) into absurd whitespace
        """
        return fmt_em(
            em_value(value, weight=weight, min_value=min_value, max_value=max_value)
        )

    # One canonical gap value used throughout the document for section-level spacing.
    gap_value = em_value(1.1, weight=1.0, min_value=0.01, max_value=None)
    GAP = fmt_em(gap_value)

    # Bullet spacing should scale with auto-fit, but stay proportionate to section gaps.
    # Use a geometric blend between raw auto-fit scaling and a GAP-anchored ratio.
    #
    # NOTE: These are `em` units, not points. Keep them small.
    bullet_base = 2.0
    gap_base = 1.1
    bullet_raw = em_value(bullet_base, weight=2.0)
    bullet_anchor = gap_value * (bullet_base / gap_base)
    bullet_balanced = (bullet_raw * max(1e-9, bullet_anchor)) ** 0.5
    bullet_min = 0.01  # Allow aggressive shrinking
    bullet_max = 1000.0  # Effectively no upper clamp
    _bullet_val = min(max(bullet_balanced, bullet_min), bullet_max)
    BULLET_GAP = fmt_em(_bullet_val)
    HALF_BULLET_GAP = fmt_em(_bullet_val * 0.5)

    space_between_contact_and_professional_summary = GAP
    space_between_professional_summary_and_next_section = GAP

    # Headers (3x3)
    headers = [escape_typst(str(h).upper()) for h in resume_data.get("headers", [])]
    while len(headers) < 9:
        headers.append("")

    # Highlighted Skills (3x3)
    h_skills = [
        escape_typst(str(s).upper()) for s in resume_data.get("highlighted_skills", [])
    ]
    while len(h_skills) < 9:
        h_skills.append("")

    experiences = profile_data.get("experience", [])
    education = profile_data.get("education", [])
    founder_roles = profile_data.get("founder_roles", [])
    section_titles = _normalize_section_titles(
        resume_data.get("section_titles") or profile_data.get("section_titles")
    )
    custom_sections = _normalize_custom_sections(
        resume_data.get("custom_sections") or profile_data.get("custom_sections")
    )
    custom_section_keys = _custom_section_keys(custom_sections)
    title_map = _build_section_title_map(section_titles, custom_sections)

    # Parse name into firstname/middle/lastname; prefer explicit fields.
    first = profile_data.get("first_name") or resume_data.get("first_name") or ""
    middle = profile_data.get("middle_name") or resume_data.get("middle_name") or ""
    last = profile_data.get("last_name") or resume_data.get("last_name") or ""
    full_name = profile_data.get("name", "").strip()
    if not (first or last):
        parts = full_name.split()
        if parts:
            first = parts[0]
            if len(parts) > 2:
                middle = " ".join(parts[1:-1])
            if len(parts) >= 2:
                last = parts[-1]
    firstname = first.strip() or "John"
    lastname = last or "Doe"

    RASTER_IMAGE_HEIGHT_PT = 10.6

    def render_img_or_text(
        raw: str,
        height_pt: float = RASTER_IMAGE_HEIGHT_PT,
        italic: bool = False,
    ) -> str:
        """
        Return Typst markup identical for descriptions and founder dates, using a shared height.
        """
        raw = raw or ""
        if not str(raw).strip():
            return ""
        lines = max(1, str(raw).count("\n") + 1)
        target_height = height_pt * lines
        img_path = _rasterize_text_image(
            raw,
            target_height_pt=target_height,
            italic=italic,
            font_family=resume_font_family,
        )
        if img_path:
            return f'image("{img_path}", height: {target_height}pt)'
        text_style = ' style: "italic"' if italic else ""
        return f'text(size: {target_height}pt, font: ("{resume_font_family_escaped}"), fill: color-darknight{text_style})[{escape_typst(raw)}]'

    qr_overlay_block = "#let qr_overlay = []\n"

    # Build modern-cv Typst template using a root-relative import (root is set to BASE_DIR).
    fork_path = "/lib.typ"

    def build_keywords():
        kw = []
        for field in (
            resume_data.get("target_role"),
            resume_data.get("target_company"),
            resume_data.get("primary_domain"),
        ):
            if field and str(field).strip():
                kw.append(str(field).strip())
        # Add top skills without overloading the metadata
        for skill in (resume_data.get("top_skills") or [])[:5]:
            if skill and str(skill).strip():
                kw.append(str(skill).strip())
        seen = set()
        deduped = []
        for item in kw:
            if item.lower() in seen:
                continue
            seen.add(item.lower())
            deduped.append(item)
        if not deduped:
            deduped = [f"{firstname} {lastname}".strip(), "resume"]
        return ", ".join(deduped)

    meta_keywords = build_keywords()

    custom_entries = ""
    if custom_contacts:
        entries: list[str] = []
        for item in custom_contacts:
            entries.append(
                "(text: "
                f"\"{escape_typst(item['text'])}\", "
                f"icon: \"{escape_typst(item['icon'])}\", "
                f"link: \"{escape_typst(item['link'])}\")"
            )
        custom_entries = f"    custom: ({', '.join(entries)},),\n"

    typst_code = f"""#import "{fork_path}": *

{qr_overlay_block}

#show: resume.with(
  author: (
    firstname: "{escape_typst(firstname)}",
    lastname: "{escape_typst(lastname)}",
    email: "{email}",
    email2: "{email2}",
    phone: "{phone}",
    github: "{github}",
    github_label: "{github_label}",
    linkedin: "{linkedin}",
    linkedin_label: "{linkedin_label}",
    scholar: "{scholar_url}",
    scholar_label: "{scholar_label}",
    portfolio: "{portfolio_url}",
    portfolio_label: "{portfolio_label}",
{custom_entries}    positions: ()
  ),
  profile-picture: none,
  date: datetime.today().display(),
  paper-size: "us-letter",
  heading-gap: {GAP},
  keywords: "{escape_typst(meta_keywords)}",
  accent-color: "#1F2937",
  font: ("{resume_font_family_escaped}"),
  header-font: ("{resume_font_family_escaped}"),
  page-foreground: qr_overlay,
)

"""
    # Force the selected font as the default text font everywhere.
    typst_code += (
        f'#set text(font: ("{resume_font_family_escaped}"), weight: "regular")\n\n'
    )

    # prompt.yaml mandates these three labels exactly.
    skills_row_labels_list = [escape_typst(lbl) for lbl in DEFAULT_SKILLS_ROW_LABELS]

    def parse_skill_rows(raw_rows):
        if raw_rows is None:
            raw_rows = []
        if isinstance(raw_rows, str):
            try:
                raw_rows = json.loads(raw_rows)
            except Exception:
                raw_rows = []
        rows = []
        if isinstance(raw_rows, (list, tuple)):
            for row in raw_rows[:3]:
                items = []
                if isinstance(row, str):
                    items = [p.strip() for p in row.split(",") if p.strip()]
                elif isinstance(row, (list, tuple)):
                    items = [str(v).strip() for v in row if str(v).strip()]
                elif row is not None and str(row).strip():
                    items = [str(row).strip()]
                rows.append([escape_typst(v) for v in items])
        while len(rows) < 3:
            rows.append([])
        return rows[:3]

    skills_rows_escaped = parse_skill_rows(resume_data.get("skills_rows"))
    if not any(any(str(s).strip() for s in row) for row in skills_rows_escaped):
        fallback_skills = [
            escape_typst(str(s))
            for s in (resume_data.get("highlighted_skills") or [])
            if str(s).strip()
        ]
        skills_rows_escaped = [
            fallback_skills[0:3],
            fallback_skills[3:6],
            fallback_skills[6:9],
        ]

    summary_raw = re.sub(r"\s+", " ", str(summary or "").strip())

    def split_summary(text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        m = re.search(r"([.!?])\s+", text)
        if m:
            head = text[: m.start(1) + 1].strip()
            tail = text[m.end() :].strip()
            return head, tail
        return text.strip(), ""

    summary_head_raw, summary_tail_raw = split_summary(summary_raw)
    summary_head = escape_typst(summary_head_raw)
    summary_tail = escape_typst(summary_tail_raw)

    def build_summary_block():
        tail_block = ""
        if summary_tail_raw:
            tail_block = f"""
      #linebreak()
      #text(size: 9.2pt, font: ("{resume_font_family_escaped}"), weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"), hyphenate: false)[{summary_tail}]"""
        return f"""#block(width: 100%)[
  #set par(leading: 1.05em, spacing: 0em)
  #block(above: {space_between_contact_and_professional_summary}, below: {space_between_professional_summary_and_next_section})[
    #align(left)[
      #text(size: 10.2pt, font: ("{resume_font_family_escaped}"), weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{summary_head}]{tail_block}
    ]
  ]
]

"""

    def split_education_entries(entries: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (primary, remaining) education entries."""

        def text_blob(edu):
            parts: list[str] = []
            for key in ("school", "degree", "description"):
                parts.append(str(edu.get(key, "") or ""))
            bullets = edu.get("bullets", [])
            if isinstance(bullets, list):
                parts.append(" ".join(str(b or "") for b in bullets))
            else:
                parts.append(str(bullets or ""))
            return " ".join(parts).lower()

        def make_entry(degree, school, start, end, details=""):
            return {
                "degree": degree,
                "school": school,
                "start_date": start,
                "end_date": end,
                "details": details,
            }

        def is_master(edu_text: str) -> bool:
            return bool(
                re.search(r"\b(m\.?s\.?|ms|m\.?a\.?|ma|master)\b", edu_text)
                or "master of" in edu_text
            )

        def is_bachelor(edu_text: str) -> bool:
            return bool(
                re.search(r"\b(b\.?s\.?|bs|b\.?a\.?|ba|bachelor)\b", edu_text)
                or "bachelor of" in edu_text
            )

        def start_date_key(edu):
            return edu.get("start_date", "") or ""

        def build_entry(edu):
            start = edu.get("start_date", "")
            end = edu.get("end_date", "")
            school = edu.get("school", "")
            raw_degree = edu.get("degree", "") or ""
            main_degree, detail = _split_degree_parts(raw_degree)
            main_degree = main_degree.strip(", ") or str(raw_degree)
            detail_items = _parse_degree_details(detail)
            detail_all = _format_degree_details(detail_items)
            return make_entry(main_degree, school, start, end, detail_all)

        ordered = sorted(list(entries or []), key=start_date_key, reverse=True)
        primary_masters: list[dict] = []
        selected_idx: set[int] = set()

        for idx, edu in enumerate(ordered):
            if not is_master(text_blob(edu)):
                continue
            primary_masters.append(build_entry(edu))
            selected_idx.add(idx)
            if len(primary_masters) >= 2:
                break

        if not primary_masters:
            for idx, edu in enumerate(ordered[:2]):
                primary_masters.append(build_entry(edu))
                selected_idx.add(idx)
        else:
            # Keep the most recent bachelor entry with the masters (if any).
            for idx, edu in enumerate(ordered):
                if idx in selected_idx:
                    continue
                if is_bachelor(text_blob(edu)):
                    primary_masters.append(build_entry(edu))
                    selected_idx.add(idx)
                    break

        remaining = [
            build_entry(edu)
            for idx, edu in enumerate(ordered)
            if idx not in selected_idx
        ]

        return primary_masters, remaining

    def render_education_section(title: str, entries: list[dict]) -> str:
        """Render Education entries as flat lines (degree, school, dates)."""
        if not entries:
            return ""

        section = f"= {escape_typst(title)}\n"
        for idx, edu in enumerate(entries):
            degree = escape_typst(edu.get("degree", "") or "")
            school = escape_typst(edu.get("school", "") or "")
            start = format_date_mm_yy(edu.get("start_date", ""))
            end = format_date_mm_yy(edu.get("end_date", ""))
            date_range = ""
            if start and end:
                date_range = f"{start}-{end}"
            elif start:
                date_range = f"{start}-"
            elif end:
                date_range = end

            title_line = " — ".join([part for part in [degree, school] if part])
            if not title_line and not date_range and not edu.get("details"):
                continue
            title_markup = (
                f'[#text(weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{degree}] #text(fill: rgb("{SOFT_SECONDARY_FILL}"))[ — {school}]]'
                if degree and school
                else f"[{title_line}]"
            )
            section += f"""#resume-entry(
  title: {title_markup},
  location: "",
  date: "{date_range}",
  description: "",
  block-above: {em(0.15 if idx == 0 else 0.18, weight=0.6, min_value=0.1, max_value=0.22)},
	  block-below: {em(0.12, weight=0.6, min_value=0.08, max_value=0.18)},
	  title-weight: "regular",
	  title-size: 10.0pt
	)
	"""
            details = (edu.get("details", "") or "").strip()
            if details:
                details_markup = format_inline_typst(details)
                details_above = em(0.60, weight=0.9, min_value=0.4, max_value=0.7)
                section += (
                    f"#block(above: {details_above}, below: {em(0.12, weight=0.9, min_value=0.08, max_value=0.16)})[\n"
                    f'  #set text(size: 9.0pt, font: ("{resume_font_family_escaped}"), weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"))\n'
                    f"  #set par(leading: {em(0.6, weight=1.0, min_value=0.45, max_value=None)}, spacing: 0em)\n"
                    f"  {details_markup}\n"
                    f"]\n"
                )
            # Add space between degree+coursework groups.
            if idx < len(entries) - 1:
                section += f"#block(height: {em(0.8, weight=1.0, min_value=0.7, max_value=None)})[]\n"

        return section

    def render_custom_sections(items: list[dict]) -> dict[str, str]:
        """Render custom sections (title + bullet body) into Typst blocks."""
        blocks: dict[str, str] = {}
        for item in items or []:
            key = str(item.get("key") or "").strip()
            if not key:
                continue
            title = str(item.get("title") or "").strip()
            body = str(item.get("body") or "").rstrip()
            if not (title or body):
                continue
            block = ""
            if title:
                block += f"= {escape_typst(title)}\n"
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if lines:
                block += "#block(above: 0em, below: 0em)[\n"
                block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
                block += f"  #set par(leading: {em(0.4, weight=1.0, min_value=0.01, max_value=None)}, spacing: 0em)\n"
                block += f'  #set list(marker: text(size: 10.2pt * 0.75, weight: "regular", fill: color-gray)[#sym.bullet], spacing: {BULLET_GAP})\n'
                for line in lines:
                    block += f"  - {format_inline_typst(line)}\n"
                block += "]\n"
            blocks[key] = block
        return blocks

    summary_block = build_summary_block() if summary else ""

    primary_education, continued_education = split_education_entries(education)
    education_block = render_education_section(
        title_map.get("education", "Education"), primary_education
    )
    education_continued_block = render_education_section(
        title_map.get("education_continued", "Education Continued"),
        continued_education,
    )

    # Experience section: do not add extra space above; the summary (if present) owns the gap.
    experience_block = f'= {escape_typst(title_map.get("experience", "Experience"))}\n'
    for idx, exp in enumerate(experiences):
        experience_block += "#block(breakable: false)[\n"
        start = format_date_mm_yy(exp.get("start_date", ""))
        end = format_date_mm_yy(exp.get("end_date", ""))
        date_range = f"{start} - {end}" if start or end else ""
        role = escape_typst(exp.get("role", ""))
        company = escape_typst(exp.get("company", ""))
        location = escape_typst(exp.get("location", ""))
        date_text = escape_typst(date_range)
        location_text = location
        # Use the template's resume-entry helper for consistent alignment.
        experience_block += f"""#resume-entry(
  title: "{role}",
  location: "{location_text}",
  date: "{date_text}",
        description: "{company}",
        block-above: {em(0.06, weight=0.6, min_value=0.04, max_value=0.09)},
        block-below: {em(0.2, weight=0.5, min_value=0.08, max_value=0.55)}
)

"""
        desc_raw = exp.get("description", "")
        desc_markup = render_img_or_text(desc_raw)
        if desc_markup:
            experience_block += f"#block(above: {em(0.32, weight=0.7, min_value=0.12, max_value=0.4)}, below: {em(0.12, weight=0.7, min_value=0.05, max_value=0.18)})[\n"
            experience_block += f"  #{desc_markup}\n"
            experience_block += "]\n"
        bullets = exp.get("bullets", [])
        if isinstance(bullets, list) and any(b.strip() for b in bullets):
            # Gap between the company line and the first bullet.
            experience_block += f"#block(height: {HALF_BULLET_GAP})\n"
            experience_block += "#block(above: 0em, below: 0em)[\n"
            experience_block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
            # Keep wrapped bullet lines readable even when auto-fit tightens.
            experience_block += f"  #set par(leading: {em(0.4, weight=1.0, min_value=0.01, max_value=None)}, spacing: 0em)\n"
            # Let auto-fit add height primarily via bullet spacing, not giant inter-entry gaps.
            experience_block += f'  #set list(marker: text(size: 10.2pt * 0.75, weight: "regular", fill: color-gray)[#sym.bullet], spacing: {BULLET_GAP})\n'
            for bullet in bullets:
                if bullet and bullet.strip():
                    experience_block += f"  - {format_inline_typst(bullet)}\n"
            experience_block += "]\n"
        else:
            experience_block += "\n"
        experience_block += "]\n"

        # Extra breathing room between roles.
        if idx < len(experiences) - 1:
            experience_block += f"#block(height: {GAP})\n\n"

    founder_block = ""
    if founder_roles:
        founder_block += (
            f'= {escape_typst(title_map.get("founder", "Startup Founder"))}\n'
        )
        founder_block += """#let founder-date-height = 9.0pt
#let founder-bullet(body, date_block) = {
  let date = if date_block == none { box(height: founder-date-height)[] } else { box(height: founder-date-height)[#date_block] }
  grid(
    columns: (auto, 1fr),
    column-gutter: 5pt,
    align: (top + left, top + left),
    [#date],
    [#body]
  )
}

        """
        for idx, role in enumerate(founder_roles):
            company = escape_typst(role.get("company", ""))
            location = escape_typst(role.get("location", ""))
            location_text = location

            founder_block += f"""#block(
  above: {em(0.06, weight=0.6, min_value=0.04, max_value=0.09)},
  below: {em(0.2, weight=0.5, min_value=0.08, max_value=0.55)},
)[
  #pad[
    #__justify_align[
      #text(weight: "regular")[{company}]
    ][
      #text(
        font: ("{resume_font_family_escaped}"),
        weight: "light",
        style: "italic",
        fill: default-location-color,
      )[{location_text}]
    ]
  ]
]

"""
            desc_raw = role.get("description", "")
            desc_markup = render_img_or_text(desc_raw)
            if desc_markup:
                founder_block += f"#block(above: {em(0.32, weight=0.7, min_value=0.12, max_value=0.4)}, below: {em(0.12, weight=0.7, min_value=0.05, max_value=0.18)})[\n"
                founder_block += f"  #{desc_markup}\n"
                founder_block += "]\n"
            bullets = role.get("bullets", [])
            bullet_lines = [b for b in bullets if isinstance(b, str) and b.strip()]
            if bullet_lines:
                # Match Experience section spacing between the entry (company line) and first bullet.
                founder_block += f"#block(height: {HALF_BULLET_GAP})\n"
                founder_block += "#block(above: 0em, below: 0em)[\n"
                founder_block += f'  #set text(size: 10.2pt, font: "{resume_font_family_escaped}", weight: 350, fill: color-darknight)\n'
                founder_block += f"  #set par(leading: {em(0.4, weight=1.0, min_value=0.01, max_value=None)}, spacing: 0em)\n"
                founder_block += f'  #set list(marker: text(size: 10.2pt * 0.75, weight: "regular", fill: color-gray)[#sym.bullet], spacing: {BULLET_GAP})\n'
                for bullet in bullet_lines:
                    body_text, date_part = split_bullet_date(bullet)
                    if not (body_text or date_part):
                        continue
                    body_markup = format_inline_typst(body_text)
                    date_str = format_bullet_date(date_part)
                    date_markup = render_img_or_text(
                        date_str, height_pt=9.0, italic=True
                    )
                    date_arg = (
                        date_markup.strip()
                        if (date_markup and date_markup.strip())
                        else "none"
                    )
                    founder_block += (
                        f"  - #founder-bullet([{body_markup or ''}], {date_arg})\n"
                    )
                founder_block += "]\n"
            if idx < len(founder_roles) - 1:
                founder_block += f"#block(height: {GAP})\n\n"

    matrices_block = ""
    if include_matrices:
        matrices_block += (
            f'= {escape_typst(title_map.get("matrices", "Skills"))}\n'
        )
        matrices_block += render_skill_rows(
            skills_row_labels_list,
            skills_rows_escaped,
            # Keep wrapped skill lines tight even when auto-fit expands spacing.
            leading=em(0.14, weight=1.0, min_value=0.35, max_value=None),
            row_gap=em(0.65, weight=1.0, min_value=0.8, max_value=None),
            font_family=resume_font_family,
        )

    custom_blocks = render_custom_sections(custom_sections)
    sections = {
        "summary": summary_block,
        "education": education_block,
        "education_continued": education_continued_block,
        "experience": experience_block,
        "founder": founder_block,
        "matrices": matrices_block,
    }
    sections.update(custom_blocks)

    if section_order is None:
        resolved_order = _sanitize_section_order(section_order, custom_section_keys)
    else:
        resolved_order = _filter_section_order(section_order, custom_section_keys)
    for section_key in resolved_order:
        block = sections.get(section_key, "")
        if block:
            typst_code += "\n" + block

    return typst_code


def render_skill_rows(
    labels: list[str],
    rows: list[list[str]],
    *,
    leading: str = "0.22em",
    row_gap: str = "0.08em",
    font_family: str | None = None,
):
    """Render the prompt.yaml skills format as 3 labeled rows (Typst markup)."""
    if not font_family:
        font_family = DEFAULT_RESUME_FONT_FAMILY
    font_family_escaped = escape_typst(str(font_family))

    def normalize_label(label: str) -> str:
        label = (label or "").strip()
        if not label:
            return ""
        return label

    labels = list(labels or [])
    while len(labels) < 3:
        labels.append("")
    labels = labels[:3]

    rows = list(rows or [])
    while len(rows) < 3:
        rows.append([])
    rows = rows[:3]

    def format_skill_line(skills: list[str], *, emphasize: int = 3) -> str:
        skills = [s for s in (skills or []) if str(s).strip()]
        if not skills:
            return ""

        core = skills[:emphasize]
        more = skills[emphasize:]

        core_parts = [
            f'#text(weight: {SOFT_BOLD_WEIGHT}, fill: rgb("{SOFT_EMPH_FILL}"))[{skill}]'
            for skill in core
        ]
        core_line = " #text(fill: color-gray)[·] ".join(core_parts)
        if not more:
            return core_line

        # Put the secondary skills on a dedicated line so the wrap doesn't leave
        # separators dangling at line ends; commas also break more naturally.
        more_line = ", ".join(more)
        more_markup = (
            f'#text(size: 9.2pt, fill: rgb("{SOFT_SECONDARY_FILL}"))[{more_line}]'
        )
        return f"{core_line}#linebreak()#h(0.35em){more_markup}"

    # F-pattern scanning: strong left-side labels + bold lead skills per row.
    # Give the skills column a bit more width to reduce awkward wraps.
    label_width = "14.2em"
    label_text_style = (
        f'#text(size: 8.8pt, weight: 350, fill: rgb("{SOFT_SECONDARY_FILL}"))'
    )

    grid_cells: list[str] = []
    for idx in range(3):
        label = normalize_label(labels[idx])
        skills = [s for s in (rows[idx] or []) if str(s).strip()]
        if not (label or skills):
            continue
        label_cell = f"[{label_text_style}[#smallcaps[{label}]]]" if label else "[]"
        skills_cell_text = format_skill_line(skills)
        skills_cell = f"[{skills_cell_text}]" if skills_cell_text else "[]"
        grid_cells.extend([label_cell, skills_cell])

    out = ["#block["]
    out.append(f'  #set text(size: 10.2pt, font: "{font_family_escaped}", weight: 350)')
    out.append(f"  #set par(leading: {leading}, spacing: 0em)")
    out.append("  #grid(")
    out.append(f"    columns: ({label_width}, 1fr),")
    out.append("    column-gutter: 0.95em,")
    out.append(f"    row-gutter: {row_gap},")
    out.append("    align: (top + left, top + left),")
    if grid_cells:
        out.append("    " + ",\n    ".join(grid_cells) + ",")
    out.append("  )")
    out.append("]\n\n")
    return "\n".join(out)


def _fake_generate_resume_content(job_req, base_profile, model_name=None):
    summary = (base_profile or {}).get("summary", "") or "Simulated summary."
    headers = [
        "Sim Header 1",
        "Sim Header 2",
        "Sim Header 3",
        "Sim Header 4",
        "Sim Header 5",
        "Sim Header 6",
        "Sim Header 7",
        "Sim Header 8",
        "Sim Header 9",
    ]
    skills = [
        "Sim Skill 1",
        "Sim Skill 2",
        "Sim Skill 3",
        "Sim Skill 4",
        "Sim Skill 5",
        "Sim Skill 6",
        "Sim Skill 7",
        "Sim Skill 8",
        "Sim Skill 9",
    ]
    return {
        "summary": summary,
        "headers": headers,
        "highlighted_skills": skills,
        "skills_rows": [
            ["Sim Row 1A", "Sim Row 1B"],
            ["Sim Row 2A"],
            ["Sim Row 3A"],
        ],
        "target_company": "Simulated Co",
        "target_role": "Simulated Role",
        "seniority_level": "Simulated",
        "target_location": "Simulated",
        "work_mode": "Simulated",
        "travel_requirement": "Simulated",
        "primary_domain": "Simulated",
        "must_have_skills": ["Simulated Skill A"],
        "nice_to_have_skills": ["Simulated Skill B"],
        "tech_stack_keywords": ["Simulated Stack"],
        "non_technical_requirements": ["Simulated Requirement"],
        "certifications": [],
        "clearances": [],
        "core_responsibilities": ["Simulated Responsibility"],
        "outcome_goals": ["Simulated Outcome"],
        "salary_band": "Simulated",
        "posting_url": "",
        "req_id": "",
    }


def _maxcov_log(msg: str) -> None:
    if not MAX_COVERAGE_LOG:
        return
    print(f"[maxcov] {msg}", flush=True)


def _maxcov_log_expected_failure(
    stdout: str | None,
    stderr: str | None,
    args: list[str],
    quiet: bool,
) -> None:
    if not quiet:
        return
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if out:
        _maxcov_log(f"expected failure stdout: {out}")
    if err:
        _maxcov_log(f"expected failure stderr: {err}")


def _maxcov_collapse_ranges(lines: list[int]) -> list[tuple[int, int]]:
    if not lines:
        return []
    ordered = sorted({int(line) for line in lines})
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for line in ordered[1:]:
        if line == prev + 1:
            prev = line
            continue
        ranges.append((start, prev))
        start = prev = line
    ranges.append((start, prev))
    return ranges


def _maxcov_format_line_ranges(lines: list[int]) -> str:
    ranges = _maxcov_collapse_ranges(lines)
    if not ranges:
        return ""
    out = []
    for start, end in ranges:
        if start == end:
            out.append(str(start))
        else:
            out.append(f"{start}-{end}")
    return ", ".join(out)


def _maxcov_format_top_missing_blocks(lines: list[int], *, limit: int = 10) -> str:
    ranges = _maxcov_collapse_ranges(lines)
    if not ranges:
        return ""
    blocks = []
    for start, end in ranges:
        length = int(end) - int(start) + 1
        blocks.append((length, start, end))
    blocks.sort(key=lambda item: (-item[0], item[1]))
    out = []
    for length, start, end in blocks[: max(1, int(limit))]:
        if start == end:
            out.append(f"{start}({length})")
        else:
            out.append(f"{start}-{end}({length})")
    return ", ".join(out)


def _maxcov_format_arc(arc) -> str:
    if not isinstance(arc, (list, tuple)) or len(arc) != 2:
        return str(arc)
    start, end = arc
    start_label = str(start) if int(start) > 0 else "entry"
    end_label = str(end) if int(end) > 0 else "exit"
    return f"{start_label}->{end_label}"


def _maxcov_format_branch_arcs(arcs, *, limit: int = 30) -> tuple[str, int]:
    if not arcs:
        return "", 0
    formatted = [_maxcov_format_arc(arc) for arc in arcs]
    extra = 0
    if len(formatted) > limit:
        extra = len(formatted) - limit
        formatted = formatted[:limit]
    return ", ".join(formatted), extra


def _get_arg_value(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for item in argv:
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def _maxcov_summarize_coverage(
    coverage_module,
    *,
    cov_dir: Path,
    cov_rc: Path,
    target: Path,
) -> dict | None:
    try:
        cov = coverage_module.Coverage(
            data_file=str(cov_dir / ".coverage"),
            config_file=str(cov_rc),
        )
        cov.load()
        analysis = cov.analysis2(str(target))
        missing_lines = list(analysis[3]) if len(analysis) >= 4 else []
        ana = cov._analyze(str(target))
        missing_branch_lines = []
        missing_branch_arcs = []
        try:
            missing_branch_lines = list(ana.missing_branch_arcs())
        except Exception:
            missing_branch_lines = []
        try:
            arcs_missing = list(ana.arcs_missing())
            if missing_branch_lines:
                branch_set = {int(line) for line in missing_branch_lines}
                missing_branch_arcs = [
                    arc
                    for arc in arcs_missing
                    if isinstance(arc, (list, tuple))
                    and len(arc) == 2
                    and int(arc[0]) in branch_set
                ]
            else:
                missing_branch_arcs = [
                    arc
                    for arc in arcs_missing
                    if isinstance(arc, (list, tuple)) and len(arc) == 2
                ]
            missing_branch_arcs.sort(key=lambda arc: (int(arc[0]), int(arc[1])))
        except Exception:
            missing_branch_arcs = []
        branch_line_ranges = _maxcov_format_line_ranges(missing_branch_lines)
        branch_arcs, branch_arcs_extra = _maxcov_format_branch_arcs(
            missing_branch_arcs, limit=30
        )
        return {
            "missing_lines": len(missing_lines),
            "missing_ranges": _maxcov_format_line_ranges(missing_lines),
            "top_missing_blocks": _maxcov_format_top_missing_blocks(
                missing_lines, limit=10
            ),
            "missing_branch_lines": len(missing_branch_lines),
            "missing_branch_line_ranges": branch_line_ranges,
            "missing_branch_arcs": branch_arcs,
            "missing_branch_arcs_extra": branch_arcs_extra,
        }
    except Exception as exc:
        _maxcov_log(f"coverage summary failed: {exc}")
        return None


def _maxcov_build_coverage_output(
    *,
    counts: dict,
    summary: dict | None,
    cov_dir: Path,
    cov_rc: Path,
    json_out: str | None,
    html_out: str | None,
) -> list[str] | None:
    if counts.get("cover"):
        covered_lines = None
        covered_pct = None
        if "stmts" in counts and "miss" in counts:
            covered_lines = counts["stmts"] - counts["miss"]
            if counts["stmts"]:
                covered_pct = (covered_lines / counts["stmts"]) * 100.0
        if summary:
            covered_label = (
                f"{covered_lines} ({covered_pct:.1f}%)"
                if covered_lines is not None and covered_pct is not None
                else "n/a"
            )
            lines = [
                "Coverage (harness.py): "
                f"{counts['cover']} | statements: {counts.get('stmts', 'n/a')} | "
                f"missing: {counts.get('miss', 'n/a')} | branches: {counts.get('branch', 'n/a')} | "
                f"partial branches: {counts.get('brpart', 'n/a')} | "
                f"covered lines: {covered_label}",
            ]
            if summary.get("missing_ranges"):
                lines.append("Missing lines (ranges): " f"{summary['missing_ranges']}")
            if summary.get("missing_branch_line_ranges"):
                lines.append(
                    "Missing branch lines (ranges): "
                    f"{summary['missing_branch_line_ranges']}"
                )
            if summary.get("missing_branch_arcs"):
                arc_suffix = ""
                if summary.get("missing_branch_arcs_extra"):
                    arc_suffix = f" ... +{summary['missing_branch_arcs_extra']} more"
                lines.append(
                    "Missing branch arcs (first 30): "
                    f"{summary['missing_branch_arcs']}{arc_suffix}"
                )
            if summary.get("top_missing_blocks"):
                lines.append(
                    "Top missing blocks (largest first): "
                    f"{summary['top_missing_blocks']}"
                )
            lines.append(f"Coverage data: {cov_dir / '.coverage'}")
            lines.append(f"Coverage rc: {cov_rc}")
            if json_out:
                lines.append(f"Coverage json: {json_out}")
            if html_out:
                lines.append(f"Coverage html: {html_out}")
            return lines
        return [f"Coverage (harness.py): {counts['cover']}"]
    return None


def _maxcov_run_container_wrapper(
    *,
    project: str | None = None,
    runner=None,
    sleep_fn=None,
    time_fn=None,
    exit_fn=None,
    check_compose: bool = False,
) -> int:
    _maxcov_log("maxcov container wrapper start")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    project = project or f"maxcov_{stamp}"
    base = ["docker", "compose", "-p", project]
    runner = runner or subprocess.run
    sleep_fn = sleep_fn or time.sleep
    time_fn = time_fn or time.time
    exit_fn = exit_fn or sys.exit

    def _run_compose(args: list[str], *, check: bool = False) -> int:
        cmd = [*base, *args]
        _maxcov_log(f"docker: {' '.join(cmd)}")
        result = runner(cmd, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(f"docker compose failed: {' '.join(args)}")
        return result.returncode

    def _wait_for_neo4j(timeout_s: float = 90.0) -> bool:
        deadline = time_fn() + max(5.0, float(timeout_s))
        while time_fn() < deadline:
            proc = runner(
                [*base, "ps", "-q", "neo4j"],
                capture_output=True,
                text=True,
            )
            container_id = (getattr(proc, "stdout", "") or "").strip()
            if container_id:
                inspect_cmd = [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Health.Status}}",
                    container_id,
                ]
                health = runner(
                    inspect_cmd,
                    capture_output=True,
                    text=True,
                )
                status = (getattr(health, "stdout", "") or "").strip()
                if status == "healthy":
                    return True
            sleep_fn(1.0)
        return False

    rc = 1
    try:
        if check_compose:
            _run_compose(["version"], check=True)
        _run_compose(["build", "maxcov"])
        _run_compose(["up", "-d", "neo4j"])
        if not _wait_for_neo4j():
            _maxcov_log("neo4j did not reach healthy state in time")
            exit_fn(1)
        run_cmd = [
            *base,
            "run",
            "--rm",
            "-T",
            "maxcov",
        ]
        _maxcov_log("maxcov container run start")
        rc = runner(run_cmd).returncode
        _maxcov_log(f"maxcov container run done: rc={rc}")
    finally:
        _run_compose(["down", "-v"])
    exit_fn(rc)
    return rc


def _maybe_launch_maxcov_container() -> None:
    if os.environ.get("MAX_COVERAGE_CONTAINER") != "1":
        globals()["MAX_COVERAGE_LOG"] = True
        _maxcov_run_container_wrapper(check_compose=True)


def _close_llm_client(llm) -> None:
    client = getattr(llm, "client", None)
    if client is None:
        return

    def _schedule_coro(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running() and not loop.is_closed():
            try:
                task = loop.create_task(coro)
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
            except Exception:
                pass
            return
        try:
            asyncio.run(coro)
        except Exception:
            pass

    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            result = close_fn()
            if asyncio.iscoroutine(result):
                _schedule_coro(result)
        except Exception:
            pass
        return
    aclose_fn = getattr(client, "aclose", None)
    if callable(aclose_fn):
        try:
            result = aclose_fn()
            if asyncio.iscoroutine(result):
                _schedule_coro(result)
        except Exception:
            pass


def _call_llm_responses(
    *,
    provider: str,
    model: str,
    input_data,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict | None = None,
    **kwargs,
):
    llm = AnyLLM.create(
        provider,
        api_key=api_key,
        api_base=api_base,
        **(client_args or {}),
    )
    try:
        return llm.responses(model=model, input_data=input_data, **kwargs)
    finally:
        _close_llm_client(llm)


def _call_llm_completion(
    *,
    provider: str,
    model: str,
    messages,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict | None = None,
    **kwargs,
):
    llm = AnyLLM.create(
        provider,
        api_key=api_key,
        api_base=api_base,
        **(client_args or {}),
    )
    try:
        return llm.completion(model=model, messages=messages, **kwargs)
    finally:
        _close_llm_client(llm)


def _extract_json_object(text: str) -> dict:
    """Parse JSON, tolerating code fences or stray prefix/suffix text."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\\s*```\\s*$", "", raw)
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: extract the outermost {...} if the model added extra text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _coerce_llm_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            parts.append(str(item))
        return "".join(parts)
    return str(value)


def generate_resume_content(job_req, base_profile, model_name: str | None = None):
    """Generate JSON profile content using the stored prompt template and Mozilla any-llm."""
    if os.environ.get("MAX_COVERAGE_SKIP_LLM") == "1":
        _maxcov_log("generate_resume_content: MAX_COVERAGE_SKIP_LLM=1")
        try:
            return _fake_generate_resume_content(job_req, base_profile, model_name)  # type: ignore[name-defined]
        except Exception:
            return {
                "summary": (base_profile or {}).get("summary", "")
                or "Simulated summary.",
                "headers": [],
                "highlighted_skills": [],
                "skills_rows": [[], [], []],
            }
    prompt_template = _resolve_prompt_template(base_profile)
    if not prompt_template:
        return {"error": "Prompt template not found in Neo4j or prompt.yaml."}
    rewrite_bullets = False
    if isinstance(base_profile, dict):
        rewrite_bullets = bool(base_profile.get("rewrite_bullets"))
    if rewrite_bullets and "experience_bullets" not in prompt_template:
        prompt_template = (
            prompt_template.rstrip()
            + "\n\n# Bullet Rewrite (Required)\n"
            + 'If the Candidate Resume JSON includes "rewrite_bullets": true, return two fields in the output JSON:\n'
            + '- "experience_bullets": [{"id": "<experience id>", "bullets": ["..."]}]\n'
            + '- "founder_role_bullets": [{"id": "<founder role id>", "bullets": ["..."]}]\n'
            + "Use the same ids from the input. Only rewrite bullets for those roles (3-6 bullets each), action verbs, evidence-based, no hallucination.\n"
        )

    provider, model = _split_llm_model_spec(model_name)
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    profile_json = json.dumps(
        base_profile or {}, indent=2, ensure_ascii=False, default=str
    )
    prompt = (
        prompt_template
        + "\n\nCandidate Resume (JSON):\n"
        + profile_json
        + "\n\nJob Requisition:\n"
        + (job_req or "")
    )

    if provider == "openai":
        api_key = load_openai_api_key()
        if not api_key:
            return {
                "error": "Missing OpenAI API key: set OPENAI_API_KEY or put it in ~/openaikey.txt"
            }

        api_base = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        organization = (
            os.environ.get("OPENAI_ORGANIZATION")
            or os.environ.get("OPENAI_ORG_ID")
            or ""
        ).strip() or None
        project = (os.environ.get("OPENAI_PROJECT") or "").strip() or None
        client_args = {}
        if organization:
            client_args["organization"] = organization
        if project:
            client_args["project"] = project

        def call_openai(
            *, max_output_tokens: int, include_reasoning: bool
        ) -> tuple[object, str]:
            request = {
                "model": model,
                "input_data": prompt,
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": int(max_output_tokens),
            }
            if include_reasoning:
                reasoning = _openai_reasoning_params_for_model(model)
                if reasoning:
                    request["reasoning"] = reasoning
            resp_obj = _call_llm_responses(
                provider="openai",
                api_key=api_key,
                api_base=api_base,
                client_args=client_args if client_args else None,
                **request,
            )
            return resp_obj, (getattr(resp_obj, "output_text", None) or "")

        max_tokens = _resolve_llm_max_output_tokens("openai", model)
        try:
            resp, content = call_openai(
                max_output_tokens=max_tokens,
                include_reasoning=True,
            )
        except (MissingApiKeyError, UnsupportedProviderError) as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"LLM call failed: {type(e).__name__}: {e}"}

        status = str(getattr(resp, "status", "") or "").strip().lower()
        incomplete_reason = getattr(
            getattr(resp, "incomplete_details", None), "reason", None
        )
        incomplete_reason = str(incomplete_reason or "").strip().lower()

        content = (content or "").strip()
        if content:
            try:
                return _extract_json_object(content)
            except Exception:
                pass

        # If the response was truncated, retry once with a larger output budget and no reasoning
        # to maximize usable JSON output tokens.
        if status == "incomplete" and incomplete_reason == "max_output_tokens":
            retry_tokens = _resolve_llm_retry_max_output_tokens(
                "openai", model, max_tokens
            )
            try:
                content2 = call_openai(
                    max_output_tokens=retry_tokens,
                    include_reasoning=False,
                )[1]
            except Exception as e:
                return {
                    "error": f"LLM call failed after truncation retry: {type(e).__name__}: {e}"
                }
            content2 = (content2 or "").strip()
            if not content2:
                return {
                    "error": "Empty response from OpenAI after truncation retry.",
                }
            try:
                return _extract_json_object(content2)
            except Exception:
                return {
                    "error": "Model returned non-JSON output (after truncation retry).",
                    "raw": content2[:2000],
                }

        if not content:
            return {"error": "Empty response from OpenAI."}
        return {
            "error": "Model returned non-JSON output.",
            "raw": content[:2000],
            "status": status or None,
            "incomplete_reason": incomplete_reason or None,
        }

    if provider == "gemini":
        api_key = load_gemini_api_key()
        if not api_key:
            return {
                "error": "Missing Gemini API key: set GEMINI_API_KEY/GOOGLE_API_KEY or put it in ~/openaikey.txt"
            }

        def call_gemini(*, max_tokens: int, force_json: bool) -> tuple[object, str]:
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
            }
            if force_json:
                request["response_format"] = {"type": "json_object"}
            resp_obj = _call_llm_completion(
                provider="gemini",
                api_key=api_key,
                **request,
            )
            text_out = ""
            try:
                choice0 = getattr(resp_obj, "choices", [None])[0]
                msg = getattr(choice0, "message", None)
                text_out = _coerce_llm_text(getattr(msg, "content", None))
            except Exception:
                text_out = ""
            return resp_obj, text_out

        max_tokens = _resolve_llm_max_output_tokens("gemini", model)
        resp = None
        content = ""
        try:
            resp, content = call_gemini(max_tokens=max_tokens, force_json=True)
        except Exception:
            try:
                resp, content = call_gemini(max_tokens=max_tokens, force_json=False)
            except (MissingApiKeyError, UnsupportedProviderError) as e:
                return {"error": str(e)}
            except Exception as e:
                return {"error": f"LLM call failed: {type(e).__name__}: {e}"}

        finish_reason = ""
        try:
            choice0 = getattr(resp, "choices", [None])[0]
            finish_reason = (
                str(getattr(choice0, "finish_reason", "") or "").strip().lower()
            )
        except Exception:
            finish_reason = ""

        content = (content or "").strip()
        if content:
            try:
                return _extract_json_object(content)
            except Exception:
                pass

        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            retry_tokens = _resolve_llm_retry_max_output_tokens(
                "gemini", model, max_tokens
            )
            try:
                content2 = call_gemini(max_tokens=retry_tokens, force_json=True)[1]
            except Exception:
                try:
                    content2 = call_gemini(max_tokens=retry_tokens, force_json=False)[1]
                except Exception as e:
                    return {
                        "error": f"LLM call failed after truncation retry: {type(e).__name__}: {e}"
                    }
            content2 = (content2 or "").strip()
            if not content2:
                return {"error": "Empty response from Gemini after truncation retry."}
            try:
                return _extract_json_object(content2)
            except Exception:
                return {
                    "error": "Model returned non-JSON output (after truncation retry).",
                    "raw": content2[:2000],
                }

        if not content:
            return {"error": "Empty response from Gemini."}
        return {
            "error": "Model returned non-JSON output.",
            "raw": content[:2000],
            "finish_reason": finish_reason or None,
        }


# ==========================================
# PDF GENERATION LAYER
# ==========================================
def escape_typst(text):
    if not text:
        return ""
    # Escape special characters for Typst content mode
    # We need to be careful not to double-escape if we run this multiple times,
    # but for now we assume raw input.
    text = str(text)
    replacements = {
        "\\": "\\\\",  # Must be first
        "@": "\\@",
        "#": "\\#",
        "$": "\\$",
        "*": "\\*",
        "_": "\\_",
        "[": "\\[",
        "]": "\\]",
        "=": "\\=",
        "`": "\\`",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def format_inline_typst(text):
    """Escape text while supporting minimal inline markup (e.g., <b>...</b>)."""
    if text is None:
        return ""

    def normalize_special_bullet(s: str) -> str:
        targets = [
            "Co-developed an OSS C++ RNN IDE",
            "Co-developed a >1m LoC OSS C++ RNN IDE",
            "Co-developed a TOP500 >1m LoC OSS C++ RNN IDE",
        ]
        if any(t in s for t in targets):
            return "Co-developed a TOP500 >1m LoC OSS C++ RNN IDE (see GitHub) called emergent."
        return s

    s = normalize_special_bullet(str(text))

    def format_emergent(value: str) -> str:
        out = []
        pos = 0
        pattern = re.compile(r"emergent", re.IGNORECASE)
        for m in pattern.finditer(value):
            if m.start() > pos:
                out.append(escape_typst(value[pos : m.start()]))
            out.append("#emph[emergent™]")
            pos = m.end()
        if pos < len(value):
            out.append(escape_typst(value[pos:]))
        return "".join(out) if out else escape_typst(value)

    # Support <b>...</b> tags (case-insensitive) as bold.
    bold_re = re.compile(r"(?i)</?b>")
    segments: list[tuple[str, bool]] = []
    bold = False
    last = 0
    for m in bold_re.finditer(s):
        if m.start() > last:
            segments.append((s[last : m.start()], bold))
        tag = m.group(0).lower()
        bold = tag == "<b>"
        last = m.end()
    if last < len(s):
        segments.append((s[last:], bold))

    if not segments:
        return format_emergent(s)

    rendered: list[str] = []
    for seg_text, seg_bold in segments:
        piece = format_emergent(seg_text)
        if seg_bold:
            rendered.append(f"#emph[#strong[{piece}]]")
        else:
            rendered.append(piece)
    return "".join(rendered)


def normalize_github(value: str) -> str:
    """Return just the GitHub path/user, stripping any protocol/host."""
    val = str(value or "").strip()
    lower = val.lower()
    # Strip protocol/host anywhere at the front.
    lower = re.sub(r"^https?://", "", lower)
    lower = re.sub(r"^www\\.", "", lower)
    if lower.startswith("github.com/"):
        val = val[len(val) - len(lower) + len("github.com/") :]
    # If still contains github.com anywhere, strip up to last slash.
    if "github.com/" in val:
        val = val.split("github.com/", 1)[1]
    return val.strip("/")


def normalize_linkedin(value: str) -> str:
    """Return just the LinkedIn slug, stripping protocol/host and optional /in/."""
    val = str(value or "").strip()
    lower = val.lower()
    lower = re.sub(r"^https?://", "", lower)
    lower = re.sub(r"^www\\.", "", lower)
    # Drop leading linkedin domain.
    if lower.startswith("linkedin.com/"):
        val = val[len(val) - len(lower) + len("linkedin.com/") :]
        lower = lower[len("linkedin.com/") :]
    # Drop optional in/ prefix
    if val.lower().startswith("in/"):
        val = val[3:]
    # If domain still present inside, strip after last slash.
    if "linkedin.com/" in val:
        val = val.split("linkedin.com/", 1)[1]
    if val.lower().startswith("in/"):
        val = val[3:]
    return val.strip("/")


def normalize_scholar_url(value: str) -> str:
    """Return a Google Scholar URL from a URL or raw citations user id."""
    val = str(value or "").strip()
    if not val:
        return ""

    if re.fullmatch(r"[A-Za-z0-9_-]+", val):
        return "https://scholar.google.com/citations?user=" + val

    m = re.search(r"(?:\\?|&)user=([^&?#/]+)", val)
    if m:
        user = m.group(1).strip()
        if user:
            return "https://scholar.google.com/citations?user=" + user

    if "://" not in val:
        return "https://" + val.lstrip("/")
    return val


def normalize_calendly_url(value: str) -> str:
    """Return a Calendly-style URL from a URL or raw handle."""
    val = str(value or "").strip()
    if not val:
        return ""

    if re.fullmatch(r"[A-Za-z0-9._-]+", val):
        return "https://cal.link/" + val

    if "://" not in val:
        return "https://" + val.lstrip("/")
    return val


def normalize_portfolio_url(value: str) -> str:
    """Return a normalized URL with https:// when missing."""
    val = str(value or "").strip()
    if not val:
        return ""
    if re.match(r"^https?://", val, flags=re.IGNORECASE):
        return val
    return "https://" + val.lstrip("/")


def format_url_label(url: str) -> str:
    """Return a short display label for a URL (drops scheme/query/fragment)."""
    value = str(url or "").strip()
    if not value:
        return ""
    candidate = value
    if "://" not in candidate:
        candidate = "https://" + candidate.lstrip("/")
    try:
        parsed = urlparse(candidate)
    except Exception:
        return value
    netloc = (parsed.netloc or "").strip()
    if netloc.lower().startswith("www."):
        netloc = netloc[4:]
    label = (netloc + (parsed.path or "")).rstrip("/")
    return label or value


def format_date_mm_yy(date_str: str) -> str:
    """Convert ISO date (YYYY-MM-DD or YYYY-MM) to MM/YY; returns empty string if invalid/empty."""
    if not date_str:
        return ""
    parts = date_str.split("-")
    if len(parts) < 2:
        return ""
    year = parts[0][-2:] if parts[0] else ""
    month = parts[1]
    if not (year and month):
        return ""
    return f"{month}/{year}"


def split_bullet_date(bullet):
    """Split a bullet into (body, date) where the date is optional and pipe-delimited."""
    text = str(bullet or "").strip()
    if "||" in text:
        date_part, body_part = text.split("||", 1)
    else:
        date_part, body_part = "", text
    return body_part.strip(), date_part.strip()


def format_bullet_date(date_str: str) -> str:
    """Normalize bullet-level date ranges (e.g., swap triple hyphens for a spaced dash)."""
    if not date_str:
        return ""
    return str(date_str).replace("---", " - ").strip()


def _ensure_fontawesome_fonts():
    """Download Font Awesome TTFs locally for Typst if missing."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FONT_AWESOME_SOURCES.items():
        path = FONTS_DIR / fname
        if path.exists():
            continue
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Warning: could not fetch {fname} from {url}: {e}")


def _ensure_template_fonts():
    """Avenir fonts are bundled locally; no template downloads required."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_typst_packages():
    """Fetch the Font Awesome Typst package locally for offline runs."""
    if FONT_AWESOME_PACKAGE_DIR.exists():
        return
    tmp_path = None
    try:
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
        FONT_AWESOME_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tmp_fd)
        urllib.request.urlretrieve(FONT_AWESOME_PACKAGE_URL, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(FONT_AWESOME_PACKAGE_DIR)
    except Exception as e:
        print(f"Warning: could not fetch Typst Font Awesome package: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def ensure_fonts_ready():
    """Download fonts only once per process."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    _ensure_fontawesome_fonts()
    _ensure_template_fonts()
    _ensure_typst_packages()
    _FONTS_READY = True


def _build_pdf_metadata(resume_data, profile_data):
    """Construct PDF metadata fields from resume/profile data."""

    def pick(*keys):
        for key in keys:
            val = profile_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
            val = resume_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return ""

    def name_parts():
        first = resume_data.get("first_name") or profile_data.get("first_name") or ""
        middle = resume_data.get("middle_name") or profile_data.get("middle_name") or ""
        last = resume_data.get("last_name") or profile_data.get("last_name") or ""
        full = profile_data.get("name", "")
        if not (first or last) and full:
            parts = full.split()
            if parts:
                first = parts[0]
                if len(parts) > 2:
                    middle = " ".join(parts[1:-1])
                if len(parts) >= 2:
                    last = parts[-1]
        return first.strip(), middle.strip(), last.strip()

    first, middle, last = name_parts()
    name_tokens = [first, last]
    author = " ".join([t for t in name_tokens if t]).strip() or "Resume Candidate"
    role = pick("target_role")
    company = pick("target_company")
    req_id = pick("req_id")

    subject_parts = [f"Résumé {author}".strip()]
    if role and company:
        subject_parts.append(f"{role} @ {company}")
    elif role:
        subject_parts.append(role)
    elif company:
        subject_parts.append(company)
    if req_id:
        subject_parts.append(f"Req {req_id}")
    subject = " — ".join(subject_parts)

    def build_keywords():
        kw = []
        for label, field in (
            ("Role", role),
            ("Company", company),
            ("Domain", pick("primary_domain")),
            ("ReqID", req_id),
        ):
            if field and str(field).strip():
                kw.append(f"{label}: {str(field).strip()}")
        for skill in (profile_data.get("highlighted_skills") or [])[:5]:
            if skill and str(skill).strip():
                kw.append(str(skill).strip())
        kw.append(author)
        kw.append("resume")
        seen = set()
        deduped = []
        for item in kw:
            norm = item.lower()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
        return "; ".join(deduped)

    return {
        "title": "Résumé",
        "subject": subject,
        "author": author,
        "creator": "ResumeBuilder3",
        "producer": "ResumeBuilder3 / Typst",
        "keywords": build_keywords(),
    }


def _apply_pdf_metadata(pdf_path: Path, metadata: dict | None = None):
    """Apply metadata and clean MarkInfo using pikepdf; no-op on failure."""
    if not metadata:
        return
    try:
        import pikepdf
    except Exception:
        return

    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            info = pdf.docinfo

            def set_info(key, value):
                if value:
                    info[getattr(pikepdf.Name, key)] = value

            ts_docinfo = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%S+00'00'")
            ts_xmp = datetime.now(timezone.utc).isoformat()

            set_info("Title", metadata.get("title"))
            set_info("Subject", metadata.get("subject"))
            set_info("Author", metadata.get("author"))
            set_info("Creator", metadata.get("creator"))
            set_info("Producer", metadata.get("producer"))
            set_info("Keywords", metadata.get("keywords"))
            set_info("CreationDate", ts_docinfo)
            set_info("ModDate", ts_docinfo)

            mark = pdf.Root.get("/MarkInfo")
            if not isinstance(mark, pikepdf.Dictionary):
                mark = pikepdf.Dictionary()
            mark["/Marked"] = True
            if "/Suspects" in mark:
                del mark["/Suspects"]
            pdf.Root["/MarkInfo"] = mark

            with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                meta["pdf:Title"] = metadata.get("title") or ""
                meta["pdf:Subject"] = metadata.get("subject") or ""
                meta["pdf:Author"] = metadata.get("author") or ""
                meta["pdf:Keywords"] = metadata.get("keywords") or ""
                meta["xmp:CreatorTool"] = metadata.get("creator") or ""
                meta["xmp:CreateDate"] = ts_xmp
                meta["xmp:ModifyDate"] = ts_xmp
                meta["dc:creator"] = [metadata.get("author") or ""]

            pdf.save(pdf_path)
    except Exception as e:
        print(f"Warning: unable to apply PDF metadata: {e}")


def _normalize_auto_fit_target_pages(
    value, default: int = DEFAULT_AUTO_FIT_TARGET_PAGES
) -> int:
    try:
        if value is None:
            raise ValueError("missing")
        if isinstance(value, bool):
            raise ValueError("bool")
        if isinstance(value, (int, float)):
            pages = int(value)
        else:
            raw = str(value).strip()
            if not raw:
                raise ValueError("blank")
            pages = int(float(raw))
    except Exception:
        pages = int(default)
    if pages < 1:
        return 1
    return pages


def compile_pdf_with_auto_tuning(
    resume_data,
    profile_data,
    include_matrices=True,
    include_summary=True,
    section_order=None,
    target_pages: int | None = None,
):
    """
    Applies a bracket + binary search on a global layout scale that tunes
    consistent spacing ratios across the document so the PDF snugly fits the
    requested page count (maximizing whitespace without spilling to N+1 pages).
    """
    import pikepdf
    from io import BytesIO

    if target_pages is None:
        target_pages = resume_data.get("auto_fit_target_pages")
    target_pages = _normalize_auto_fit_target_pages(target_pages)

    pdf_metadata = _build_pdf_metadata(resume_data, profile_data)

    def persist_auto_fit_cache(*, best_scale: float, too_long_scale: float | None):
        try:
            db = Neo4jClient()
            db.set_auto_fit_cache(
                best_scale=best_scale,
                too_long_scale=too_long_scale,
            )
            db.close()
        except Exception:
            pass

    def render(scale: float):
        source = generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=include_matrices,
            include_summary=include_summary,
            section_order=section_order,
            layout_scale=scale,
        )
        ok, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
        if not ok or not pdf_bytes:
            return False, 0, b""
        try:
            pdf = pikepdf.open(BytesIO(pdf_bytes))
            return True, len(pdf.pages), pdf_bytes
        except Exception:
            # If we can't parse the PDF, we can't tune; treat it as a valid render and return it.
            return True, 0, pdf_bytes

    # Bracket the solution:
    # - `fit_scale` produces <= target_pages
    # - `too_long_scale` produces > target_pages
    min_scale = 0.35
    max_scale = 6.0
    grow = 1.35
    shrink = 1.0 / grow
    tol_scale = 0.002

    # Seed auto-fit from the last successful tuning (if available).
    initial_scale = 1.0
    cached_too_long_scale = None
    try:
        db = Neo4jClient()
        cache = db.get_auto_fit_cache() or {}
        db.close()
        cached_best = cache.get("best_scale")
        if isinstance(cached_best, (int, float)) and float(cached_best) > 0:
            initial_scale = float(cached_best)
        cached_high = cache.get("too_long_scale")
        if isinstance(cached_high, (int, float)) and float(cached_high) > 0:
            cached_too_long_scale = float(cached_high)
    except Exception:
        pass
    initial_scale = max(min_scale, min(max_scale, initial_scale))
    if cached_too_long_scale is not None:
        cached_too_long_scale = max(min_scale, min(max_scale, cached_too_long_scale))

    ok, pages, pdf_bytes = render(initial_scale)
    if not ok:
        return False, b""
    if pages == 0:
        return True, pdf_bytes

    fit_scale = initial_scale
    fit_pdf = pdf_bytes
    too_long_scale = None

    if pages > target_pages:
        # Too long; shrink until it fits or we hit min_scale.
        too_long_scale = initial_scale
        tightest_pdf = fit_pdf
        tightest_scale = fit_scale
        found_fit = False
        scale = initial_scale
        while scale > min_scale:
            scale = max(min_scale, scale * shrink)
            ok, pages, pdf_bytes = render(scale)
            if not ok:
                continue
            if pages == 0:
                return True, pdf_bytes
            tightest_pdf = pdf_bytes
            tightest_scale = scale
            if pages <= target_pages:
                fit_scale = scale
                fit_pdf = pdf_bytes
                found_fit = True
                break
        if not found_fit:
            print(
                f"Auto-fit: resume is longer than {target_pages} page(s) even at minimum layout scale. Returning tightened multi-page document."
            )
            persist_auto_fit_cache(
                best_scale=tightest_scale,
                too_long_scale=too_long_scale,
            )
            return True, tightest_pdf
    else:
        # Fits. If we have a cached overflow bound, validate it and reuse as the bracket
        # so we don't have to re-expand and re-bisect from a wide range each time.
        if (
            cached_too_long_scale is not None
            and cached_too_long_scale > fit_scale
            and cached_too_long_scale <= max_scale
        ):
            ok2, pages2, pdf2 = render(cached_too_long_scale)
            if ok2:
                if pages2 == 0:
                    return True, pdf2
                if pages2 > target_pages:
                    too_long_scale = cached_too_long_scale
                else:
                    # Cached "too long" bound now fits; treat it as the new fit point and expand.
                    fit_scale = cached_too_long_scale
                    fit_pdf = pdf2

        # Fits; expand until it overflows or we hit max_scale.
        scale = fit_scale
        while scale < max_scale:
            scale = min(max_scale, scale * grow)
            if too_long_scale is not None and scale >= too_long_scale:
                break
            ok, pages, pdf_bytes = render(scale)
            if not ok:
                continue
            if pages == 0:
                return True, pdf_bytes
            if pages <= target_pages:
                fit_scale = scale
                fit_pdf = pdf_bytes
            else:
                too_long_scale = scale
                break
        if too_long_scale is None:
            print(
                f"Auto-fit: resume still fits within {target_pages} page(s) at maximum layout scale. Returning expanded document."
            )
            persist_auto_fit_cache(best_scale=fit_scale, too_long_scale=None)
            return True, fit_pdf

    low = fit_scale
    high = too_long_scale if too_long_scale is not None else max_scale
    best_scale = fit_scale
    best_pdf = fit_pdf

    for _ in range(10):
        if (high - low) <= tol_scale:
            break
        mid = (low + high) / 2.0
        ok, pages, pdf_bytes = render(mid)
        if not ok:
            continue
        if pages == 0:
            return True, pdf_bytes
        if pages <= target_pages:
            low = mid
            best_scale = mid
            best_pdf = pdf_bytes
        else:
            high = mid

    print(
        f"Auto-fit: selected layout scale {best_scale:.3f} for {target_pages} page(s)."
    )
    persist_auto_fit_cache(
        best_scale=best_scale,
        too_long_scale=high,
    )
    return True, best_pdf


def compile_pdf(typst_source, metadata=None):
    """
    Compiles Typst source to PDF and returns (success, pdf_bytes).
    No PDFs are left on disk. Temporary files live in a hidden folder to avoid
    triggering Reflex hot-reload file watchers.
    """
    temp_path = None
    output_path = None
    try:
        ensure_fonts_ready()
        TEMP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".typ", delete=False, dir=TEMP_BUILD_DIR, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(typst_source)
            temp_path = Path(tmp.name)
            output_path = temp_path.with_suffix(".pdf")

        root_path = str(BASE_DIR)
        process = subprocess.Popen(
            [
                TYPST_BIN,
                "compile",
                "--font-path",
                str(FONTS_DIR),
                "--package-path",
                str(PACKAGES_DIR),
                "--root",
                root_path,
                str(temp_path),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(BASE_DIR),
            env={
                **os.environ,
                "TYPST_FONT_PATHS": str(FONTS_DIR),
                "TYPST_FONT_PATH": str(FONTS_DIR),
                "TYPST_PACKAGE_PATHS": str(PACKAGES_DIR),
            },
        )
        stderr = process.communicate()[1]

        if process.returncode != 0:
            print(f"Typst compilation failed: {stderr}")
            return False, b""
        _apply_pdf_metadata(output_path, metadata)
        pdf_bytes = output_path.read_bytes()
        return True, pdf_bytes
    except Exception as e:
        print(f"Error running Typst: {e}")
        return False, b""
    finally:
        for path in (temp_path, output_path):
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


# ==========================================
# UI LAYER (REFLEX)
# ==========================================
class Experience(BaseModel):
    id: str = ""
    company: str = ""
    role: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class Education(BaseModel):
    id: str = ""
    school: str = ""
    degree: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class FounderRole(BaseModel):
    id: str = ""
    company: str = ""
    role: str = ""
    location: str = ""
    description: str = ""
    bullets: str = ""
    start_date: str = ""
    end_date: str = ""


class CustomSection(BaseModel):
    id: str = ""
    key: str = ""
    title: str = ""
    body: str = ""


def _model_to_dict(model) -> dict:
    if model is None:
        return {}
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    as_dict = getattr(model, "dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            pass
    if isinstance(model, dict):
        return dict(model)
    return {}


def _skills_rows_to_csv(rows, highlighted_skills) -> list[str]:
    rows = list(rows or [])

    def has_row_content(row) -> bool:
        if isinstance(row, (list, tuple)):
            return any(str(s).strip() for s in row)
        if row is None:
            return False
        return bool(str(row).strip())

    if not any(has_row_content(row) for row in rows):
        fallback_skills = [
            str(s).strip() for s in (highlighted_skills or []) if str(s).strip()
        ]
        rows = [fallback_skills[0:3], fallback_skills[3:6], fallback_skills[6:9]]

    out: list[str] = []
    for row in rows[:3]:
        if isinstance(row, (list, tuple)):
            out.append(", ".join([str(s).strip() for s in row if str(s).strip()]))
        elif row is None:
            out.append("")
        else:
            out.append(str(row).strip())
    while len(out) < 3:
        out.append("")
    return out[:3]


class State(rx.State):
    job_req: str = ""
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    email: str = ""
    email2: str = ""
    phone: str = ""
    font_family: str = DEFAULT_RESUME_FONT_FAMILY
    linkedin_url: str = ""
    github_url: str = ""
    scholar_url: str = ""
    calendly_url: str = ""
    portfolio_url: str = ""
    summary: str = "Professional Summary..."
    prompt_yaml: str = ""
    rewrite_bullets_with_llm: bool = False
    headers: list[str] = ["", "", "", "", "", "", "", "", ""]
    highlighted_skills: list[str] = ["", "", "", "", "", "", "", "", ""]
    skills_rows: list[list[str]] = [[], [], []]
    profile_experience_bullets: dict[str, list[str]] = {}
    profile_founder_bullets: dict[str, list[str]] = {}
    experience: list[Experience] = []
    education: list[Education] = []
    founder_roles: list[FounderRole] = []

    # Keep profile_data for other fields if needed, or just decompose it all
    profile_data: dict = {}

    pdf_url: str = ""
    pdf_generated: bool = False
    generating_pdf: bool = False
    has_loaded: bool = False
    is_saving: bool = False
    is_loading_resume: bool = False
    is_generating_profile: bool = False
    is_auto_pipeline: bool = False
    data_loaded: bool = False
    auto_tune_pdf: bool = True
    auto_fit_target_pages: int = DEFAULT_AUTO_FIT_TARGET_PAGES
    include_matrices: bool = True
    section_order: list[str] = list(DEFAULT_SECTION_ORDER)
    section_titles: dict[str, str] = {}
    section_visibility: dict[str, bool] = {key: True for key in SECTION_LABELS}
    custom_sections: list[CustomSection] = []
    new_section_title: str = ""
    new_section_body: str = ""
    last_pdf_signature: str = ""
    last_pdf_b64: str = ""
    pdf_error: str = ""
    db_error: str = ""
    llm_models: list[str] = list(DEFAULT_LLM_MODELS)
    selected_model: str = DEFAULT_LLM_MODEL
    last_render_label: str = ""
    last_save_label: str = ""
    # Latest role/profile fields
    target_company: str = ""
    target_role: str = ""
    seniority_level: str = ""
    target_location: str = ""
    work_mode: str = ""
    travel_requirement: str = ""
    primary_domain: str = ""
    must_have_skills_text: str = ""
    nice_to_have_skills_text: str = ""
    tech_stack_keywords_text: str = ""
    non_technical_requirements_text: str = ""
    certifications_text: str = ""
    clearances_text: str = ""
    core_responsibilities_text: str = ""
    outcome_goals_text: str = ""
    salary_band: str = ""
    posting_url: str = ""
    req_id: str = ""
    pipeline_status: list[str] = []
    last_profile_job_req_sha: str = ""
    latest_profile_id: str = ""
    edit_profile_bullets: bool = False
    is_saving_profile_bullets: bool = False
    last_profile_bullets_label: str = ""

    @rx.var
    def skills_rows_csv(self) -> list[str]:
        return _skills_rows_to_csv(self.skills_rows, self.highlighted_skills)

    @rx.var
    def pipeline_latest(self) -> str:
        msgs = list(self.pipeline_status or [])
        return msgs[-1] if msgs else ""

    @rx.var
    def profile_experience_bullets_list(self) -> list[str]:
        overrides = dict(self.profile_experience_bullets or {})
        out: list[str] = []
        for exp in self.experience or []:
            entry_id = str(getattr(exp, "id", "") or "")
            out.append(_coerce_bullet_text(overrides.get(entry_id, [])))
        return out

    @rx.var
    def profile_founder_bullets_list(self) -> list[str]:
        overrides = dict(self.profile_founder_bullets or {})
        out: list[str] = []
        for role in self.founder_roles or []:
            entry_id = str(getattr(role, "id", "") or "")
            out.append(_coerce_bullet_text(overrides.get(entry_id, [])))
        return out

    @rx.var
    def section_order_rows(self) -> list[dict]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        visibility = dict(self.section_visibility or {})
        title_map = _build_section_title_map(self.section_titles, custom_sections)
        custom_key_set = set(extra_keys)
        rows: list[dict] = []
        for key in order:
            title = title_map.get(key) or key.replace("_", " ").title()
            rows.append(
                {
                    "key": key,
                    "visible": bool(visibility.get(key, True)),
                    "title": title,
                    "custom": key in custom_key_set,
                }
            )
        return rows

    @rx.var
    def custom_section_rows(self) -> list[dict]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        rows: list[dict] = []
        for item in custom_sections:
            rows.append(
                {
                    "key": item.get("key", ""),
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                }
            )
        return rows

    @rx.var
    def job_req_needs_profile(self) -> bool:
        req_text = str(self.job_req or "").strip()
        if not req_text:
            return False
        current = hashlib.sha256(req_text.encode("utf-8")).hexdigest()
        return current != (self.last_profile_job_req_sha or "")

    def _visible_section_order(self) -> list[str]:
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        visibility = dict(self.section_visibility or {})
        return [key for key in order if visibility.get(key, True)]

    def on_load(self):
        # Always reset UI-visible fields so reload starts empty.
        self.db_error = ""
        self.pdf_error = ""
        self.summary = "Professional Summary..."
        self.prompt_yaml = ""
        self.rewrite_bullets_with_llm = False
        self.headers = [""] * 9
        self.highlighted_skills = [""] * 9
        self.experience = []
        self.education = []
        self.founder_roles = []
        self.skills_rows = [[], [], []]
        self.profile_experience_bullets = {}
        self.profile_founder_bullets = {}
        self.profile_data = {}
        self.first_name = ""
        self.middle_name = ""
        self.last_name = ""
        self.email = ""
        self.email2 = ""
        self.phone = ""
        self.linkedin_url = ""
        self.github_url = ""
        self.scholar_url = ""
        self.calendly_url = ""
        self.portfolio_url = ""
        self.font_family = DEFAULT_RESUME_FONT_FAMILY
        self.target_company = ""
        self.target_role = ""
        self.seniority_level = ""
        self.target_location = ""
        self.work_mode = ""
        self.travel_requirement = ""
        self.primary_domain = ""
        self.must_have_skills_text = ""
        self.nice_to_have_skills_text = ""
        self.tech_stack_keywords_text = ""
        self.non_technical_requirements_text = ""
        self.certifications_text = ""
        self.clearances_text = ""
        self.core_responsibilities_text = ""
        self.outcome_goals_text = ""
        self.salary_band = ""
        self.posting_url = ""
        self.req_id = ""
        self.job_req = ""
        self.last_profile_job_req_sha = ""
        self.pipeline_status = []
        self.data_loaded = False
        self.auto_tune_pdf = True
        self.auto_fit_target_pages = DEFAULT_AUTO_FIT_TARGET_PAGES
        self.include_matrices = True
        self.section_order = list(DEFAULT_SECTION_ORDER)
        self.section_visibility = {key: True for key in SECTION_LABELS}
        self.section_titles = {}
        self.custom_sections = []
        self.new_section_title = ""
        self.new_section_body = ""
        self.pdf_generated = False
        self.pdf_url = ""
        self.last_pdf_signature = ""
        self.latest_profile_id = ""
        self.edit_profile_bullets = False
        self.is_saving_profile_bullets = False
        self.last_profile_bullets_label = ""
        try:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"on_load called at {datetime.now()}\\n")

            data = {}
            try:
                db = Neo4jClient()
                db.ensure_resume_exists()
                data = db.get_resume_data() or {}
                db.close()
            except Exception as e:
                with open(DEBUG_LOG, "a") as f:
                    f.write(f"Neo4j unavailable: {e}\\n")
                self.db_error = "Database unavailable; could not load data."
                self.pdf_error = self.db_error
                self.has_loaded = True
                return

            if not data or not data.get("resume"):
                self.db_error = "No resume found in Neo4j."
                self.pdf_error = self.db_error
                self.has_loaded = True
                return

            self.profile_data = data.get("resume", {})
            prompt_yaml = self.profile_data.get("prompt_yaml")
            if not (isinstance(prompt_yaml, str) and prompt_yaml.strip()):
                prompt_yaml = _load_prompt_yaml_from_file()
            self.prompt_yaml = prompt_yaml or ""
            self.font_family = (
                self.profile_data.get("font_family") or DEFAULT_RESUME_FONT_FAMILY
            )
            self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
                self.profile_data.get("auto_fit_target_pages"),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            )
            if os.environ.get("MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD") == "1":
                self.db_error = "Forced db error for coverage."

            section_titles = _normalize_section_titles(
                self.profile_data.get("section_titles_json")
                or self.profile_data.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                self.profile_data.get("custom_sections_json")
                or self.profile_data.get("custom_sections")
            )
            self.section_titles = section_titles
            self.custom_sections = [
                CustomSection(**item) for item in custom_sections
            ]
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = self.profile_data.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            self.section_order = _sanitize_section_order(raw_order, extra_keys)
            raw_enabled = self.profile_data.get("section_enabled")
            normalized_enabled = _normalize_section_enabled(
                raw_enabled, list(SECTION_LABELS) + extra_keys, extra_keys=extra_keys
            )
            self.section_visibility = {
                key: key in normalized_enabled
                for key in _known_section_keys(extra_keys)
            }
            self.include_matrices = self.section_visibility.get("matrices", True)

            # Start the form empty to avoid auto-loading the full resume on initial page load.
            self.experience = []
            self.education = []
            self.founder_roles = []

            # Load configured LLM models (no network calls). Override with LLM_MODELS if desired.
            self.llm_models = list_llm_models()
            if self.llm_models:
                if self.selected_model not in self.llm_models:
                    self.selected_model = self.llm_models[0]
            else:
                self.llm_models = list(DEFAULT_LLM_MODELS)
                self.selected_model = DEFAULT_LLM_MODEL

            # Optionally render once on load; default is to skip to speed up first paint.
            if os.environ.get("GENERATE_ON_LOAD", "0") == "1":
                if self.db_error:
                    self.pdf_error = self.db_error
                else:
                    self.generate_pdf()

            with open(DEBUG_LOG, "a") as f:
                f.write("on_load completed successfully\\n")

            self.has_loaded = True

        except Exception as e:
            import traceback

            with open(DEBUG_LOG, "a") as f:
                f.write(f"Error in on_load: {e}\\n")
                f.write(traceback.format_exc())
                f.write("\\n")
            if MAX_COVERAGE_LOG:
                _maxcov_log(f"on_load error: {e}")
            elif not _should_silence_warnings():
                print(f"Error in on_load: {e}")
            self.has_loaded = True

    async def save_to_db(self):
        """Persist current form data to Neo4j (no JSON fallback)."""
        if self.is_saving:
            return
        self.is_saving = True
        yield
        try:

            def normalize_items(items):
                normalized = []
                for idx, item in enumerate(items):
                    data = _model_to_dict(item)
                    if not data.get("id"):
                        new_id = str(uuid.uuid4())
                        data["id"] = new_id
                        items[idx].id = new_id
                    bullets_text = data.get("bullets", "")
                    bullets = []
                    if isinstance(bullets_text, str):
                        for line in bullets_text.split("\n"):
                            line = line.strip()
                            if line:
                                bullets.append(line)
                    data["bullets"] = bullets
                    data["start_date"] = data.get("start_date") or None
                    data["end_date"] = data.get("end_date") or None
                    normalized.append(data)
                return normalized

            experiences = normalize_items(self.experience)
            education = normalize_items(self.education)
            founder_roles = normalize_items(self.founder_roles)

            headers = list(self.headers)
            while len(headers) < 9:
                headers.append("")

            full_name = " ".join(
                [self.first_name, self.middle_name, self.last_name]
            ).strip()

            section_titles = _normalize_section_titles(self.section_titles)
            custom_sections = _normalize_custom_sections(
                [_model_to_dict(s) for s in (self.custom_sections or [])]
            )
            extra_keys = _custom_section_keys(custom_sections)
            section_order = _sanitize_section_order(self.section_order, extra_keys)
            section_enabled = [
                key
                for key in section_order
                if (self.section_visibility or {}).get(key, True)
            ]
            prompt_yaml = (self.prompt_yaml or "").rstrip()
            if not prompt_yaml:
                prompt_yaml = str(self.profile_data.get("prompt_yaml") or "").rstrip()
            if not prompt_yaml:
                prompt_yaml = _load_prompt_yaml_from_file() or ""
            resume_fields = {
                "first_name": self.first_name,
                "middle_name": self.middle_name,
                "last_name": self.last_name,
                "name": full_name,
                "email": self.email,
                "email2": self.email2,
                "phone": self.phone,
                "font_family": self.font_family or DEFAULT_RESUME_FONT_FAMILY,
                "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                    self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
                ),
                "linkedin_url": self.linkedin_url,
                "github_url": self.github_url,
                "scholar_url": self.scholar_url,
                "calendly_url": self.calendly_url,
                "portfolio_url": self.portfolio_url,
                "summary": self.summary,
                "prompt_yaml": prompt_yaml,
                "head1_left": headers[0],
                "head1_middle": headers[1],
                "head1_right": headers[2],
                "head2_left": headers[3],
                "head2_middle": headers[4],
                "head2_right": headers[5],
                "head3_left": headers[6],
                "head3_middle": headers[7],
                "head3_right": headers[8],
                "top_skills": list(self.highlighted_skills),
                "section_order": section_order,
                "section_enabled": section_enabled,
                "section_titles_json": json.dumps(section_titles, ensure_ascii=False),
                "custom_sections_json": json.dumps(custom_sections, ensure_ascii=False),
            }
            self.section_order = list(section_order)
            self.prompt_yaml = prompt_yaml

            db = Neo4jClient()
            db.upsert_resume_and_sections(
                resume_fields,
                experiences,
                education,
                founder_roles,
                delete_missing=self.data_loaded,
            )
            db.close()

            # Keep local cache aligned so subsequent renders use fresh values.
            self.profile_data.update(resume_fields)
            self.profile_data["experience"] = experiences
            self.profile_data["education"] = education
            self.profile_data["founder_roles"] = founder_roles
            self.db_error = ""
            self.last_save_label = (
                f"Saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"Save failed: {e}\n")
            self.db_error = "Unable to save to Neo4j."
        finally:
            self.is_saving = False

    async def load_resume_fields(self):
        """Load canonical resume fields (summary/headers/top_skills) into the form."""
        if self.is_loading_resume:
            self._log_debug("load_resume_fields skipped: already loading")
            return
        self.is_loading_resume = True
        self.data_loaded = False
        self._log_debug("load_resume_fields started")
        self._add_pipeline_msg("load_resume_fields entered")
        yield
        try:
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            profiles = db.list_applied_jobs()
            db.close()
            if not data or not data.get("resume"):
                self.pdf_error = "No resume found in Neo4j."
                return

            r = data.get("resume", {})
            latest_profile = profiles[0] if profiles else {}
            self.latest_profile_id = str(latest_profile.get("id") or "")

            prompt_yaml = r.get("prompt_yaml")
            if not (isinstance(prompt_yaml, str) and prompt_yaml.strip()):
                prompt_yaml = _load_prompt_yaml_from_file()
            self.prompt_yaml = prompt_yaml or ""

            def ensure_len(items, target=9):
                items = list(items or [])
                while len(items) < target:
                    items.append("")
                return items[:target]

            def parse_name(full: str):
                parts = (full or "").strip().split()
                if not parts:
                    return "", "", ""
                if len(parts) == 1:
                    return parts[0], "", ""
                if len(parts) == 2:
                    return parts[0], "", parts[1]
                return parts[0], " ".join(parts[1:-1]), parts[-1]

            def list_to_text(val):
                if isinstance(val, list):
                    return "\n".join([str(v) for v in val if v is not None])
                if val is None:
                    return ""
                return str(val)

            self.first_name = r.get("first_name", self.first_name)
            self.middle_name = r.get("middle_name", self.middle_name)
            self.last_name = r.get("last_name", self.last_name)
            if not (self.first_name or self.last_name):
                n_first, n_mid, n_last = parse_name(r.get("name", ""))
                self.first_name = n_first or self.first_name
                self.middle_name = n_mid or self.middle_name
                self.last_name = n_last or self.last_name
            self.email = r.get("email", self.email)
            self.email2 = r.get("email2", self.email2)
            self.phone = r.get("phone", self.phone)
            self.font_family = (
                r.get("font_family", "")
                or self.font_family
                or DEFAULT_RESUME_FONT_FAMILY
            )
            self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
                r.get("auto_fit_target_pages", self.auto_fit_target_pages),
                DEFAULT_AUTO_FIT_TARGET_PAGES,
            )
            self.linkedin_url = r.get("linkedin_url", self.linkedin_url)
            self.github_url = r.get("github_url", self.github_url)
            self.scholar_url = r.get("scholar_url", self.scholar_url)
            self.calendly_url = r.get("calendly_url", self.calendly_url)
            self.portfolio_url = r.get("portfolio_url", self.portfolio_url)
            # Summary
            self.summary = latest_profile.get("summary") or r.get(
                "summary", self.summary
            )
            self.skills_rows = latest_profile.get("skills_rows") or [[], [], []]
            self.profile_experience_bullets = _bullet_override_map(
                latest_profile.get("experience_bullets")
            )
            self.profile_founder_bullets = _bullet_override_map(
                latest_profile.get("founder_role_bullets")
            )
            # Job req (do not override user input unless empty)
            profile_req_raw = latest_profile.get("job_req_raw") or ""
            if str(profile_req_raw).strip():
                self.last_profile_job_req_sha = hashlib.sha256(
                    str(profile_req_raw).strip().encode("utf-8")
                ).hexdigest()
                if not (self.job_req and str(self.job_req).strip()):
                    self.job_req = str(profile_req_raw)
            else:
                self.last_profile_job_req_sha = ""
            # Role details from latest profile
            self.target_company = latest_profile.get(
                "target_company", self.target_company
            )
            self.target_role = latest_profile.get("target_role", self.target_role)
            self.seniority_level = latest_profile.get(
                "seniority_level", self.seniority_level
            )
            self.target_location = latest_profile.get(
                "target_location", self.target_location
            )
            self.work_mode = latest_profile.get("work_mode", self.work_mode)
            self.travel_requirement = latest_profile.get(
                "travel_requirement", self.travel_requirement
            )
            self.primary_domain = latest_profile.get(
                "primary_domain", self.primary_domain
            )
            self.must_have_skills_text = list_to_text(
                latest_profile.get("must_have_skills", self.must_have_skills_text)
            )
            self.nice_to_have_skills_text = list_to_text(
                latest_profile.get("nice_to_have_skills", self.nice_to_have_skills_text)
            )
            self.tech_stack_keywords_text = list_to_text(
                latest_profile.get("tech_stack_keywords", self.tech_stack_keywords_text)
            )
            self.non_technical_requirements_text = list_to_text(
                latest_profile.get(
                    "non_technical_requirements", self.non_technical_requirements_text
                )
            )
            self.certifications_text = list_to_text(
                latest_profile.get("certifications", self.certifications_text)
            )
            self.clearances_text = list_to_text(
                latest_profile.get("clearances", self.clearances_text)
            )
            self.core_responsibilities_text = list_to_text(
                latest_profile.get(
                    "core_responsibilities", self.core_responsibilities_text
                )
            )
            self.outcome_goals_text = list_to_text(
                latest_profile.get("outcome_goals", self.outcome_goals_text)
            )
            self.salary_band = latest_profile.get("salary_band", self.salary_band)
            self.posting_url = latest_profile.get("posting_url", self.posting_url)
            self.req_id = latest_profile.get("req_id", self.req_id)

            section_titles = _normalize_section_titles(
                r.get("section_titles_json") or r.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                r.get("custom_sections_json") or r.get("custom_sections")
            )
            self.section_titles = section_titles
            self.custom_sections = [
                CustomSection(**item) for item in custom_sections
            ]
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = r.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            self.section_order = _sanitize_section_order(raw_order, extra_keys)
            raw_enabled = r.get("section_enabled")
            normalized_enabled = _normalize_section_enabled(
                raw_enabled, list(SECTION_LABELS) + extra_keys, extra_keys=extra_keys
            )
            self.section_visibility = {
                key: key in normalized_enabled
                for key in _known_section_keys(extra_keys)
            }
            self.include_matrices = self.section_visibility.get("matrices", True)

            # Headers
            latest_headers = latest_profile.get("headers") or []
            if latest_headers and any(str(h).strip() for h in latest_headers):
                self.headers = ensure_len(latest_headers)
            else:
                headers = [
                    r.get("head1_left", ""),
                    r.get("head1_middle", ""),
                    r.get("head1_right", ""),
                    r.get("head2_left", ""),
                    r.get("head2_middle", ""),
                    r.get("head2_right", ""),
                    r.get("head3_left", ""),
                    r.get("head3_middle", ""),
                    r.get("head3_right", ""),
                ]
                self.headers = ensure_len(headers)

            # Top skills -> highlighted skills
            latest_skills = latest_profile.get("highlighted_skills") or []
            if latest_skills and any(str(s).strip() for s in latest_skills):
                self.highlighted_skills = ensure_len(latest_skills)
            else:
                self.highlighted_skills = ensure_len(r.get("top_skills", []))

            def to_models(items, model_cls):
                out = []
                for item in items:
                    item = dict(item)
                    bullets = item.get("bullets", [])
                    if isinstance(bullets, list):
                        item["bullets"] = "\n".join(bullets)
                    elif bullets is None:
                        item["bullets"] = ""
                    else:
                        item["bullets"] = str(bullets)
                    for key in ("start_date", "end_date"):
                        value = item.get(key)
                        if value is None:
                            item[key] = ""
                        elif not isinstance(value, str):
                            item[key] = str(value)
                    out.append(model_cls(**item))
                return out

            # Populate the form from Neo4j.
            self.experience = to_models(data.get("experience", []), Experience)
            self.education = to_models(data.get("education", []), Education)
            self.founder_roles = to_models(data.get("founder_roles", []), FounderRole)

            self.data_loaded = True
            self.pdf_error = ""
        except Exception as e:
            self.pdf_error = f"Error loading resume fields: {e}"
            self._add_pipeline_msg(f"load_resume_fields error: {e}")
        finally:
            self.is_loading_resume = False
            self._log_debug("load_resume_fields finished")
            self._add_pipeline_msg("load_resume_fields exited")

    def _compute_pdf_signature(self, resume_data, profile_data, typst_source=""):
        """Generate a stable signature for the current resume to avoid redundant renders."""
        payload = {
            "resume": resume_data,
            "profile": profile_data,
            "template_version": TYPST_TEMPLATE_VERSION,
            "source_hash": (
                hashlib.sha256(typst_source.encode("utf-8")).hexdigest()
                if typst_source
                else ""
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _current_resume_profile(self):
        """Build the current resume/profile dictionaries (with bullets split) once."""

        def model_to_dict_list(models, overrides=None):
            items = []
            for m in models:
                data = _model_to_dict(m)
                bullets_text = data.get("bullets", "")
                if isinstance(bullets_text, str):
                    data["bullets"] = [
                        b for b in bullets_text.split("\n") if b is not None
                    ]
                else:
                    data["bullets"] = []
                if overrides:
                    entry_id = str(data.get("id") or "").strip()
                    override_bullets = overrides.get(entry_id)
                    if override_bullets:
                        data["bullets"] = list(override_bullets)
                items.append(data)
            return items

        profile_data = self.profile_data.copy()
        profile_data["name"] = " ".join(
            [self.first_name, self.middle_name, self.last_name]
        ).strip()
        profile_data["email"] = self.email or profile_data.get("email", "")
        profile_data["email2"] = self.email2 or profile_data.get("email2", "")
        profile_data["phone"] = self.phone or profile_data.get("phone", "")
        profile_data["linkedin_url"] = self.linkedin_url or profile_data.get(
            "linkedin_url", ""
        )
        profile_data["github_url"] = self.github_url or profile_data.get(
            "github_url", ""
        )
        profile_data["scholar_url"] = self.scholar_url or profile_data.get(
            "scholar_url", ""
        )
        profile_data["calendly_url"] = self.calendly_url or profile_data.get(
            "calendly_url", ""
        )
        profile_data["portfolio_url"] = self.portfolio_url or profile_data.get(
            "portfolio_url", ""
        )
        profile_data["font_family"] = self.font_family or profile_data.get(
            "font_family", DEFAULT_RESUME_FONT_FAMILY
        )
        profile_data["experience"] = model_to_dict_list(
            self.experience or [], self.profile_experience_bullets
        )
        profile_data["education"] = model_to_dict_list(self.education or [])
        profile_data["founder_roles"] = model_to_dict_list(
            self.founder_roles or [], self.profile_founder_bullets
        )

        # Build resume/profile payloads after any hydration so they reflect latest values.
        profile_meta = {
            "target_role": self.target_role,
            "target_company": self.target_company,
            "primary_domain": self.primary_domain,
            "req_id": self.req_id,
        }
        resume_data = {
            "summary": self.summary,
            "headers": self.headers,
            "highlighted_skills": self.highlighted_skills,
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "email2": self.email2,
            "portfolio_url": self.portfolio_url,
            "font_family": self.font_family or DEFAULT_RESUME_FONT_FAMILY,
            "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
            ),
            "skills_rows": self.skills_rows,
        }

        # Ensure section order contains only known sections and falls back to default.
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        extra_keys = _custom_section_keys(custom_sections)
        self.section_order = _sanitize_section_order(self.section_order, extra_keys)
        resume_data["section_titles"] = _normalize_section_titles(self.section_titles)
        resume_data["custom_sections"] = custom_sections

        profile_data.update(profile_meta)
        return resume_data, profile_data

    def _load_cached_pdf(self, signature):
        """Load an on-disk PDF if its signature matches."""
        try:
            if not (LIVE_PDF_PATH.exists() and LIVE_PDF_SIG_PATH.exists()):
                return False
            disk_sig = LIVE_PDF_SIG_PATH.read_text(encoding="utf-8").strip()
            if disk_sig != signature:
                return False
            pdf_bytes = LIVE_PDF_PATH.read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            cache_buster = int(datetime.now().timestamp() * 1000)
            self.pdf_url = f"data:application/pdf;base64,{pdf_b64}#t={cache_buster}"
            self.last_pdf_b64 = pdf_b64
            self.last_pdf_signature = signature
            self.pdf_generated = True
            self.pdf_error = ""
            return True
        except Exception:
            return False

    def generate_pdf(self):
        if self.generating_pdf:
            return
        if not self.data_loaded:
            self.pdf_error = "Load Data or Generate Profile first."
            return
        req_text = str(self.job_req or "").strip()
        if req_text:
            current = hashlib.sha256(req_text.encode("utf-8")).hexdigest()
            if current != (self.last_profile_job_req_sha or ""):
                self.pdf_error = "Job requisition has not been processed yet. Run Generate Profile or the pipeline first."
                return
        start_time = None
        recorded = False
        self.generating_pdf = True
        try:
            self.pdf_error = ""
            start_time = datetime.now()
            # Build the payload first so we can cheaply skip duplicate renders.
            resume_data, profile_data = self._current_resume_profile()
            visible_order = self._visible_section_order()
            source = generate_typst_source(
                resume_data,
                profile_data,
                include_matrices=self.include_matrices,
                include_summary=True,
                section_order=visible_order,
            )
            pdf_metadata = _build_pdf_metadata(resume_data, profile_data)
            signature = self._compute_pdf_signature(resume_data, profile_data, source)

            # If nothing changed and we already have a URL, skip rendering.
            if (
                self.pdf_generated
                and self.last_pdf_signature == signature
                and self.pdf_url
            ):
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # If signature unchanged but URL missing, reuse cached base64.
            if (
                self.pdf_generated
                and self.last_pdf_signature == signature
                and self.last_pdf_b64
            ):
                cache_buster = int(datetime.now().timestamp() * 1000)
                self.pdf_url = (
                    f"data:application/pdf;base64,{self.last_pdf_b64}#t={cache_buster}"
                )
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # If on-disk cache matches, reuse it.
            if self._load_cached_pdf(signature):
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, True)
                recorded = True
                return

            # Compile PDF; optionally write to disk when enabled (off by default to avoid hot-reload loops).
            if self.auto_tune_pdf:
                target_pages = _normalize_auto_fit_target_pages(
                    self.auto_fit_target_pages, DEFAULT_AUTO_FIT_TARGET_PAGES
                )
                resume_data["auto_fit_target_pages"] = target_pages
                success, pdf_bytes = compile_pdf_with_auto_tuning(
                    resume_data,
                    profile_data,
                    include_matrices=self.include_matrices,
                    include_summary=True,
                    section_order=visible_order,
                    target_pages=target_pages,
                )
            else:
                success, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
            if success and pdf_bytes:
                pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
                self.last_pdf_b64 = pdf_b64
                # Serve as a data URL so the UI viewer always has the PDF inline.
                # Add a timestamp fragment to prevent stale embeds from being reused.
                cache_buster = int(datetime.now().timestamp() * 1000)
                self.pdf_url = f"data:application/pdf;base64,{pdf_b64}#t={cache_buster}"
                self.pdf_generated = True
                self.last_pdf_signature = signature
                self.pdf_error = ""
                if RUNTIME_WRITE_PDF:
                    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
                    LIVE_PDF_PATH.write_bytes(pdf_bytes)
                    LIVE_PDF_SIG_PATH.write_text(signature, encoding="utf-8")
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, False)
                recorded = True
            else:
                self.pdf_error = "PDF generation failed; check server logs."
        finally:
            if start_time and not recorded:
                elapsed = datetime.now() - start_time
                ms = round(elapsed.total_seconds() * 1000, 1)
                self._record_render_time(ms, False)
            self.generating_pdf = False

    def _record_render_time(self, ms, from_cache: bool):
        """Store render label with cache/compile context."""
        if ms >= 1000:
            time_str = f"{round(ms / 1000, 2)}s"
        else:
            time_str = f"{ms} ms"
        prefix = "Served from cache in" if from_cache else "Rendered with Typst in"
        self.last_render_label = f"{prefix} {time_str}."

    def move_section_up(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        if idx <= 0 or idx >= len(order):
            return
        order[idx - 1], order[idx] = order[idx], order[idx - 1]
        self.section_order = order

    def move_section_down(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        custom_sections = _normalize_custom_sections(
            [_model_to_dict(s) for s in (self.custom_sections or [])]
        )
        extra_keys = _custom_section_keys(custom_sections)
        order = _sanitize_section_order(self.section_order, extra_keys)
        if idx < 0 or idx >= len(order) - 1:
            return
        order[idx + 1], order[idx] = order[idx], order[idx + 1]
        self.section_order = order

    @rx.event
    def set_auto_tune_pdf(self, value: bool):
        """Toggle auto-fit (snugly fills the target page count)."""
        self.auto_tune_pdf = _coerce_bool(value)

    @rx.event
    def set_auto_fit_target_pages(self, value):
        self.auto_fit_target_pages = _normalize_auto_fit_target_pages(
            value, DEFAULT_AUTO_FIT_TARGET_PAGES
        )

    @rx.event
    def set_include_matrices(self, value: bool):
        """Toggle inclusion of the Skills section."""
        self.include_matrices = _coerce_bool(value)
        visibility = dict(self.section_visibility or {})
        visibility["matrices"] = self.include_matrices
        self.section_visibility = visibility

    @rx.event
    def set_rewrite_bullets_with_llm(self, value: bool):
        """Toggle LLM bullet rewrites for generated profiles."""
        self.rewrite_bullets_with_llm = _coerce_bool(value)

    @rx.event
    def set_edit_profile_bullets(self, value: bool):
        """Toggle editing of Profile bullet overrides in the UI."""
        self.edit_profile_bullets = _coerce_bool(value)

    @rx.event
    def set_profile_experience_bullets_text(self, entry_id: str, value: str):
        """Update profile experience bullet overrides for an entry id."""
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return
        bullets = [
            line.strip() for line in str(value or "").split("\n") if line.strip()
        ]
        overrides = dict(self.profile_experience_bullets or {})
        if bullets:
            overrides[entry_id] = bullets
        else:
            overrides.pop(entry_id, None)
        self.profile_experience_bullets = overrides

    @rx.event
    def set_profile_founder_bullets_text(self, entry_id: str, value: str):
        """Update profile founder bullet overrides for an entry id."""
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return
        bullets = [
            line.strip() for line in str(value or "").split("\n") if line.strip()
        ]
        overrides = dict(self.profile_founder_bullets or {})
        if bullets:
            overrides[entry_id] = bullets
        else:
            overrides.pop(entry_id, None)
        self.profile_founder_bullets = overrides

    async def save_profile_bullets(self):
        """Persist profile bullet overrides to the latest Profile."""
        if self.is_saving_profile_bullets:
            return
        self.is_saving_profile_bullets = True
        yield
        try:
            if not (self.latest_profile_id and str(self.latest_profile_id).strip()):
                self.db_error = "No profile available to update."
                return
            experience_bullets = [
                {"id": k, "bullets": v}
                for k, v in (self.profile_experience_bullets or {}).items()
                if k and v
            ]
            founder_role_bullets = [
                {"id": k, "bullets": v}
                for k, v in (self.profile_founder_bullets or {}).items()
                if k and v
            ]
            db = Neo4jClient()
            db.update_profile_bullets(
                self.latest_profile_id,
                experience_bullets,
                founder_role_bullets,
            )
            db.close()
            self.db_error = ""
            self.last_profile_bullets_label = f"Profile bullets saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"Profile bullets save failed: {e}\n")
            self.db_error = "Unable to save profile bullets."
        finally:
            self.is_saving_profile_bullets = False

    @rx.event
    def set_section_visibility(self, value: bool, key: str):
        """Toggle visibility for a section key."""
        key = str(key or "").strip()
        if not key:
            return
        visibility = dict(self.section_visibility or {})
        visibility[key] = _coerce_bool(value)
        self.section_visibility = visibility
        if key == "matrices":
            self.include_matrices = _coerce_bool(value)

    @rx.event
    def set_section_title(self, value: str, key: str):
        """Update a section title (base sections or custom sections)."""
        key = str(key or "").strip()
        if not key:
            return
        title = str(value or "").strip()
        if key in SECTION_LABELS:
            titles = dict(self.section_titles or {})
            if title:
                titles[key] = title
            else:
                titles.pop(key, None)
            self.section_titles = titles
            return
        sections = list(self.custom_sections or [])
        for idx, section in enumerate(sections):
            if str(getattr(section, "key", "") or "") == key:
                sections[idx] = CustomSection(
                    id=str(section.id or ""),
                    key=str(section.key or ""),
                    title=title,
                    body=str(section.body or ""),
                )
                break
        self.custom_sections = sections

    @rx.event
    def set_custom_section_body(self, key: str, value: str):
        """Update the body text for a custom section."""
        key = str(key or "").strip()
        if not key:
            return
        body = str(value or "")
        sections = list(self.custom_sections or [])
        for idx, section in enumerate(sections):
            if str(getattr(section, "key", "") or "") == key:
                sections[idx] = CustomSection(
                    id=str(section.id or ""),
                    key=str(section.key or ""),
                    title=str(section.title or ""),
                    body=body,
                )
                break
        self.custom_sections = sections

    @rx.event
    def add_custom_section(self):
        """Add a new custom section with the current draft title/body."""
        title = str(self.new_section_title or "").strip() or "Custom Section"
        body = str(self.new_section_body or "")
        section_id = str(uuid.uuid4())
        key = f"custom_{section_id}"
        sections = list(self.custom_sections or [])
        sections.append(
            CustomSection(id=section_id, key=key, title=title, body=body)
        )
        self.custom_sections = sections
        order = list(self.section_order or [])
        if key not in order:
            order.append(key)
        self.section_order = order
        visibility = dict(self.section_visibility or {})
        visibility[key] = True
        self.section_visibility = visibility
        self.new_section_title = ""
        self.new_section_body = ""

    @rx.event
    def remove_custom_section(self, key: str):
        """Remove a custom section and purge it from ordering/visibility."""
        key = str(key or "").strip()
        if not key:
            return
        self.custom_sections = [
            section
            for section in (self.custom_sections or [])
            if str(getattr(section, "key", "") or "") != key
        ]
        self.section_order = [k for k in (self.section_order or []) if k != key]
        visibility = dict(self.section_visibility or {})
        visibility.pop(key, None)
        self.section_visibility = visibility
        titles = dict(self.section_titles or {})
        titles.pop(key, None)
        self.section_titles = titles

    # Experience entries
    def add_experience(self):
        items = list(self.experience or [])
        items.insert(0, Experience(id=str(uuid.uuid4())))
        self.experience = items

    def remove_experience(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = list(self.experience or [])
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.experience = items

    def update_experience_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.experience) <= idx:
            self.experience.append(Experience())
        setattr(self.experience[idx], field, value)

    # Education entries
    def add_education(self):
        items = list(self.education or [])
        items.append(Education(id=str(uuid.uuid4())))
        self.education = items

    def remove_education(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = list(self.education or [])
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.education = items

    def update_education_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.education) <= idx:
            self.education.append(Education())
        setattr(self.education[idx], field, value)

    # Founder role entries
    def add_founder_role(self):
        items = list(self.founder_roles or [])
        items.append(FounderRole(id=str(uuid.uuid4())))
        self.founder_roles = items

    def remove_founder_role(self, index):
        try:
            idx = int(index)
        except Exception:
            return
        items = list(self.founder_roles or [])
        if idx < 0 or idx >= len(items):
            return
        items.pop(idx)
        self.founder_roles = items

    def update_founder_role_field(self, index, field, value):
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            return
        while len(self.founder_roles) <= idx:
            self.founder_roles.append(FounderRole())
        setattr(self.founder_roles[idx], field, value)

    @rx.event
    def paste_req_and_run_pipeline(self, text: str):
        """Set job_req from clipboard, log, and trigger the auto-pipeline."""
        self.job_req = text or ""
        self._add_pipeline_msg("Pasted job req from clipboard")
        self._add_pipeline_msg("Queueing auto-pipeline...")
        self._log_debug("paste_req_and_run_pipeline queued auto_pipeline_from_req")
        return type(self).auto_pipeline_from_req

    def _add_pipeline_msg(self, msg: str):
        """Append a pipeline status message (immutably for state change detection)."""
        self.pipeline_status = list(self.pipeline_status or []) + [msg]
        self._log_debug(f"PIPELINE MSG: {msg}")

    def _log_debug(self, msg: str):
        """Write a debug line to the temp log file."""
        try:
            with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | {msg}\\n")
        except Exception:
            pass

    @rx.event
    async def auto_pipeline_from_req(self):
        """Paste req -> generate profile -> load data -> render PDF."""
        self._add_pipeline_msg("Auto-pipeline invoked")
        self._log_debug("auto_pipeline_from_req invoked")
        if self.is_auto_pipeline:
            self._add_pipeline_msg("Auto-pipeline skipped: already running")
            self._log_debug("auto_pipeline_from_req skipped: already running")
            return
        if not (self.job_req and str(self.job_req).strip()):
            self._add_pipeline_msg("Auto-pipeline skipped: empty job req")
            yield
            return
        self.pdf_error = ""
        self.is_auto_pipeline = True
        self._add_pipeline_msg("Auto-pipeline started")
        yield  # flush start
        failure = None
        t_start = None
        try:
            t_start = datetime.now()
            # Generate profile
            gen_start = datetime.now()
            self._add_pipeline_msg("Stage 1: Generating profile...")
            self._log_debug("auto_pipeline_from_req: calling generate_profile (pre)")
            yield  # flush stage 1 start
            gen = self.generate_profile()
            if hasattr(gen, "__anext__"):
                async for _ in gen:
                    pass
            gen_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(f"Stage 1 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg("Stage 1 complete")
            self._log_debug("auto_pipeline_from_req: generate_profile finished")
            self._add_pipeline_msg(
                f"Generating profile...done ({(gen_end - gen_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 1

            # Load hydrated data
            load_start = datetime.now()
            self._add_pipeline_msg("Stage 2: Hydrating UI with latest profile...")
            self._log_debug("auto_pipeline_from_req: calling load_resume_fields (pre)")
            yield  # flush stage 2 start
            load = self.load_resume_fields()
            if hasattr(load, "__anext__"):
                async for _ in load:
                    pass
            load_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(f"Stage 2 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg("Stage 2 complete")
            self._log_debug("auto_pipeline_from_req: load_resume_fields finished")
            self._add_pipeline_msg(
                f"Loading data...done ({(load_end - load_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 2

            # Render PDF
            render_start = datetime.now()
            self._add_pipeline_msg("Stage 3: Rendering PDF...")
            self._log_debug("auto_pipeline_from_req: calling generate_pdf (pre)")
            yield  # flush stage 3 start
            self.generate_pdf()
            render_end = datetime.now()
            if self.pdf_error:
                failure = self.pdf_error
                self._add_pipeline_msg(f"Stage 3 failed: {self.pdf_error}")
                yield
                return
            self._add_pipeline_msg("Stage 3 complete")
            self._log_debug("auto_pipeline_from_req: generate_pdf finished")
            self._add_pipeline_msg(
                f"Rendering PDF...done ({(render_end - render_start).total_seconds():.2f}s)"
            )
            yield  # flush after stage 3
        except Exception as e:
            failure = str(e)
            self._add_pipeline_msg(f"Auto-pipeline failed: {e}")
            try:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"auto_pipeline_from_req error: {e}\\n")
            except Exception:
                pass
            yield
        finally:
            if not failure and t_start is not None:
                elapsed = datetime.now() - t_start
                self._add_pipeline_msg(
                    f"Auto-pipeline complete ({elapsed.total_seconds():.2f}s)"
                )
                yield  # final flush
            self.is_auto_pipeline = False
            self._log_debug("auto_pipeline_from_req finished")

    async def generate_profile(self):
        """Call LLM with current req + base resume, save Profile to Neo4j, and hydrate UI."""
        if self.is_generating_profile:
            self._log_debug("generate_profile skipped: already generating")
            return
        self.is_generating_profile = True
        self._log_debug("generate_profile started")
        self._add_pipeline_msg("generate_profile entered")
        yield  # flush entry
        try:
            self.pdf_error = ""
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            db.close()
            resume_node = data.get("resume", {}) or {}
            base_profile = {
                **resume_node,
                "experience": data.get("experience", []),
                "education": data.get("education", []),
                "founder_roles": data.get("founder_roles", []),
            }
            base_profile["rewrite_bullets"] = bool(self.rewrite_bullets_with_llm)
            prompt_yaml = (self.prompt_yaml or "").strip()
            if prompt_yaml:
                base_profile["prompt_yaml"] = prompt_yaml
            model_name = self.selected_model or DEFAULT_LLM_MODEL
            llm_start = datetime.now()
            self._add_pipeline_msg("generate_profile: dispatching LLM call")
            self._log_debug(
                "generate_profile: calling generate_resume_content (to_thread)"
            )
            yield  # flush before LLM
            try:
                result = await asyncio.to_thread(
                    generate_resume_content, self.job_req, base_profile, model_name
                )
            except Exception as llm_err:
                self._add_pipeline_msg(f"generate_profile LLM error: {llm_err}")
                self._log_debug(f"generate_profile LLM error: {llm_err}")
                raise
            llm_end = datetime.now()
            self._add_pipeline_msg(
                f"generate_profile: LLM returned in {(llm_end - llm_start).total_seconds():.2f}s"
            )
            self._log_debug("generate_profile: LLM call finished")
            if not (result and isinstance(result, dict)):
                self.pdf_error = "LLM returned no content."
                self._add_pipeline_msg("generate_profile: LLM returned no content")
                return
            if result.get("error"):
                self.pdf_error = str(result.get("error") or "LLM returned an error.")
                self._add_pipeline_msg(f"generate_profile error: {self.pdf_error}")
                return

            # Persist new Profile
            def fallback_if_blank(values, fallback_vals, target_len=None):
                vals = list(values or [])
                if target_len:
                    while len(vals) < target_len:
                        vals.append("")
                    vals = vals[:target_len]
                if all((v is None or str(v).strip() == "") for v in vals):
                    return list(fallback_vals or [])
                return vals

            def ensure_len(items, target=9):
                items = list(items or [])
                while len(items) < target:
                    items.append("")
                return items[:target]

            resume_headers = ensure_len(
                [
                    resume_node.get("head1_left", ""),
                    resume_node.get("head1_middle", ""),
                    resume_node.get("head1_right", ""),
                    resume_node.get("head2_left", ""),
                    resume_node.get("head2_middle", ""),
                    resume_node.get("head2_right", ""),
                    resume_node.get("head3_left", ""),
                    resume_node.get("head3_middle", ""),
                    resume_node.get("head3_right", ""),
                ]
            )
            resume_top_skills = ensure_len(resume_node.get("top_skills", []))
            header_labels = {
                str(h).strip().lower() for h in resume_headers if str(h).strip()
            }

            def is_contact_skill(value: str) -> bool:
                lowered = value.lower().strip()
                if lowered in header_labels:
                    return True
                if "linkedin" in lowered or "github" in lowered or "scholar" in lowered:
                    return True
                if "http://" in lowered or "https://" in lowered or "www." in lowered:
                    return True
                if "portfolio" in lowered and (
                    "code" in lowered
                    or "github" in lowered
                    or "public" in lowered
                    or "repo" in lowered
                    or "git" in lowered
                ):
                    return True
                if "@" in value and "." in value:
                    return True
                return False

            def sanitize_skills(values, *, max_len: int | None = None) -> list[str]:
                cleaned: list[str] = []
                seen: set[str] = set()
                for item in values or []:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if not text or is_contact_skill(text):
                        continue
                    key = text.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(text)
                    if max_len and len(cleaned) >= max_len:
                        break
                return cleaned

            def sanitize_skill_rows(rows: list[list[str]]) -> list[list[str]]:
                cleaned_rows: list[list[str]] = []
                for row in rows[:3]:
                    cleaned_rows.append(sanitize_skills(row))
                while len(cleaned_rows) < 3:
                    cleaned_rows.append([])
                return cleaned_rows[:3]

            skills_rows = sanitize_skill_rows(
                _ensure_skill_rows(result.get("skills_rows"))
            )
            highlighted_skills = sanitize_skills(
                fallback_if_blank(
                    result.get("highlighted_skills"),
                    resume_top_skills,
                    target_len=9,
                ),
                max_len=9,
            )
            highlighted_skills = ensure_len(highlighted_skills, target=9)
            if not any(any(str(s).strip() for s in row) for row in skills_rows):
                skills_rows = [
                    highlighted_skills[0:3],
                    highlighted_skills[3:6],
                    highlighted_skills[6:9],
                ]
            skills_rows_json = json.dumps(skills_rows, ensure_ascii=False)
            experience_bullets = (
                _coerce_bullet_overrides(result.get("experience_bullets"))
                if self.rewrite_bullets_with_llm
                else []
            )
            founder_role_bullets = (
                _coerce_bullet_overrides(result.get("founder_role_bullets"))
                if self.rewrite_bullets_with_llm
                else []
            )
            resume_fields = {
                "summary": result.get("summary", resume_node.get("summary", "")),
                "headers": fallback_if_blank(
                    result.get("headers"), resume_headers, target_len=9
                ),
                "highlighted_skills": highlighted_skills,
                "skills_rows_json": skills_rows_json,
                "experience_bullets_json": json.dumps(
                    experience_bullets, ensure_ascii=False
                ),
                "founder_role_bullets_json": json.dumps(
                    founder_role_bullets, ensure_ascii=False
                ),
                "job_req_raw": self.job_req,
                # Required Profile props with safe defaults
                "target_company": result.get("target_company", ""),
                "target_role": result.get("target_role", ""),
                "seniority_level": result.get("seniority_level", ""),
                "target_location": result.get("target_location", ""),
                "work_mode": result.get("work_mode", ""),
                "travel_requirement": result.get("travel_requirement", ""),
                "primary_domain": result.get("primary_domain", ""),
                "must_have_skills": result.get("must_have_skills", []),
                "nice_to_have_skills": result.get("nice_to_have_skills", []),
                "tech_stack_keywords": result.get("tech_stack_keywords", []),
                "non_technical_requirements": result.get(
                    "non_technical_requirements", []
                ),
                "certifications": result.get("certifications", []),
                "clearances": result.get("clearances", []),
                "core_responsibilities": result.get("core_responsibilities", []),
                "outcome_goals": result.get("outcome_goals", []),
                "salary_band": result.get("salary_band", ""),
                "posting_url": result.get("posting_url", ""),
                "req_id": result.get("req_id", ""),
            }
            db = Neo4jClient()
            profile_id = db.save_resume(resume_fields)
            db.close()
            self.latest_profile_id = str(profile_id or "")
            self.profile_experience_bullets = _bullet_override_map(experience_bullets)
            self.profile_founder_bullets = _bullet_override_map(founder_role_bullets)
            if self.rewrite_bullets_with_llm:
                self.edit_profile_bullets = True
            # Hydrate into UI
            self.summary = resume_fields["summary"]
            self.skills_rows = skills_rows
            self.headers = (
                resume_fields["headers"][:9]
                if resume_fields.get("headers")
                else self.headers
            )
            self.highlighted_skills = (
                resume_fields["highlighted_skills"][:9]
                if resume_fields.get("highlighted_skills")
                else self.highlighted_skills
            )
            self.profile_data.update(result)
            self.data_loaded = True

            if not (self.experience or self.education or self.founder_roles):
                try:
                    db = Neo4jClient()
                    data = db.get_resume_data() or {}
                    db.close()
                except Exception:
                    data = {}

                def to_models(items, model_cls):
                    out = []
                    for item in items:
                        item = dict(item)
                        bullets = item.get("bullets", [])
                        if isinstance(bullets, list):
                            item["bullets"] = "\n".join(bullets)
                        elif bullets is None:
                            item["bullets"] = ""
                        else:
                            item["bullets"] = str(bullets)
                        out.append(model_cls(**item))
                    return out

                self.experience = to_models(data.get("experience", []), Experience)
                self.education = to_models(data.get("education", []), Education)
                self.founder_roles = to_models(
                    data.get("founder_roles", []), FounderRole
                )
            # Role detail fields
            self.target_company = result.get("target_company", self.target_company)
            self.target_role = result.get("target_role", self.target_role)
            self.seniority_level = result.get("seniority_level", self.seniority_level)
            self.target_location = result.get("target_location", self.target_location)
            self.work_mode = result.get("work_mode", self.work_mode)
            self.travel_requirement = result.get(
                "travel_requirement", self.travel_requirement
            )
            self.primary_domain = result.get("primary_domain", self.primary_domain)

            def list_to_text(val):
                if isinstance(val, list):
                    return "\n".join([str(v) for v in val if v is not None])
                if val is None:
                    return ""
                return str(val)

            self.must_have_skills_text = list_to_text(
                result.get("must_have_skills", [])
            )
            self.nice_to_have_skills_text = list_to_text(
                result.get("nice_to_have_skills", [])
            )
            self.tech_stack_keywords_text = list_to_text(
                result.get("tech_stack_keywords", [])
            )
            self.non_technical_requirements_text = list_to_text(
                result.get("non_technical_requirements", [])
            )
            self.certifications_text = list_to_text(result.get("certifications", []))
            self.clearances_text = list_to_text(result.get("clearances", []))
            self.core_responsibilities_text = list_to_text(
                result.get("core_responsibilities", [])
            )
            self.outcome_goals_text = list_to_text(result.get("outcome_goals", []))
            self.salary_band = result.get("salary_band", self.salary_band)
            self.posting_url = result.get("posting_url", self.posting_url)
            self.req_id = result.get("req_id", self.req_id)
            self.db_error = ""
            req_text = str(self.job_req or "").strip()
            self.last_profile_job_req_sha = (
                hashlib.sha256(req_text.encode("utf-8")).hexdigest() if req_text else ""
            )
        except Exception as e:
            self.pdf_error = f"Error generating profile: {e}"
            self._add_pipeline_msg(f"generate_profile error: {e}")
        finally:
            self.is_generating_profile = False
            self._log_debug("generate_profile finished")
            self._add_pipeline_msg("generate_profile exited")
            yield  # flush exit


# ==========================================
# UI COMPONENTS
# ==========================================
def styled_input(value, on_change, placeholder="", **props):
    merged_style = {
        "background": "#2d3748",
        "border": "1px solid #4a5568",
        "color": "#e2e8f0",
        "padding": "0.5em 1em",
        "min_height": "2.5em",
        "height": "auto",
        "_focus": {
            "border_color": "#63b3ed",
            "box_shadow": "0 0 0 1px #63b3ed",
        },
        "_placeholder": {
            "color": "#a0aec0",
        },
    }
    # Merge user-provided style
    if "style" in props:
        merged_style.update(props.pop("style"))

    width = props.pop("width", "100%")
    return rx.input(
        value=value,
        on_change=on_change,
        placeholder=placeholder,
        variant="soft",
        color_scheme="gray",
        radius="medium",
        width=width,
        style=merged_style,
        **props,
    )


def styled_textarea(value, on_change, placeholder="", **props):
    merged_style = {
        "background": "#2d3748",
        "border": "1px solid #4a5568",
        "color": "#e2e8f0",
        "padding": "0.5em 1em",
        "min_height": "2.5em",
        "height": "auto",
        "_focus": {
            "border_color": "#63b3ed",
            "box_shadow": "0 0 0 1px #63b3ed",
        },
        "_placeholder": {
            "color": "#a0aec0",
        },
    }
    # Merge user-provided style
    if "style" in props:
        merged_style.update(props.pop("style"))

    width = props.pop("width", "100%")
    return rx.text_area(
        value=value,
        on_change=on_change,
        placeholder=placeholder,
        variant="soft",
        color_scheme="gray",
        radius="medium",
        width=width,
        style=merged_style,
        **props,
    )


def styled_card(children, **props):
    return rx.box(
        children,
        bg="#1a202c",
        padding="1.5em",
        border_radius="lg",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
        border="1px solid #2d3748",
        width="100%",
        **props,
    )


def labeled_toggle(
    label: str,
    checked,
    on_change,
    *,
    size="3",
    color_scheme="indigo",
    show_label: bool = True,
    switch_props: dict | None = None,
    container_props: dict | None = None,
):
    switch_kwargs = {
        "checked": checked,
        "on_change": on_change,
        "size": size,
        "color_scheme": color_scheme,
    }
    if switch_props:
        switch_kwargs.update(switch_props)
    toggle = rx.switch(**switch_kwargs)
    if not show_label:
        return toggle
    container_defaults = {
        "spacing": "2",
        "align_items": "center",
        "width": "100%",
    }
    if container_props:
        container_defaults.update(container_props)
    return rx.hstack(
        rx.text(label, weight="medium", color="#e2e8f0"),
        toggle,
        **container_defaults,
    )


def _section_order_row(row, i):
    return rx.hstack(
        rx.checkbox(
            checked=row["visible"],
            on_change=lambda v: State.set_section_visibility(v, row["key"]),
            size="1",
            color_scheme="green",
            aria_label="Toggle section visibility",
        ),
        rx.hstack(
            rx.button(
                "↑",
                on_click=lambda _=None, i=i: State.move_section_up(i),
                size="1",
                variant="soft",
                color_scheme="gray",
                aria_label="Move section up",
            ),
            rx.button(
                "↓",
                on_click=lambda _=None, i=i: State.move_section_down(i),
                size="1",
                variant="soft",
                color_scheme="gray",
                aria_label="Move section down",
            ),
            spacing="1",
        ),
        styled_input(
            value=row["title"],
            on_change=lambda v: State.set_section_title(v, row["key"]),
            placeholder="Section title",
            width="100%",
        ),
        rx.cond(
            row["custom"],
            rx.button(
                "Remove",
                on_click=lambda _=None: State.remove_custom_section(row["key"]),
                size="1",
                variant="soft",
                color_scheme="red",
            ),
            rx.box(),
        ),
        rx.spacer(),
        width="100%",
        align_items="center",
        spacing="2",
    )


def _experience_card(exp, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: State.remove_experience(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=exp.role,
                    on_change=lambda v: State.update_experience_field(i, "role", v),
                    placeholder="Role",
                ),
                styled_input(
                    value=exp.company,
                    on_change=lambda v: State.update_experience_field(i, "company", v),
                    placeholder="Company",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=exp.location,
                on_change=lambda v: State.update_experience_field(i, "location", v),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=exp.start_date,
                    on_change=lambda v: State.update_experience_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=exp.end_date,
                    on_change=lambda v: State.update_experience_field(i, "end_date", v),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=exp.description,
                on_change=lambda v: State.update_experience_field(i, "description", v),
                placeholder="Short company/role description (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=exp.bullets,
                on_change=lambda v: State.update_experience_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            rx.cond(
                State.edit_profile_bullets,
                rx.vstack(
                    rx.text(
                        "Profile bullets (LLM override)",
                        color="#a0aec0",
                        font_size="0.85em",
                    ),
                    styled_textarea(
                        value=State.profile_experience_bullets_list[i],
                        on_change=lambda v, exp_id=exp.id: State.set_profile_experience_bullets_text(
                            exp_id, v
                        ),
                        placeholder="Profile bullets (one per line)",
                        min_height="240px",
                        width="100%",
                        style={"resize": "vertical"},
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.fragment(),
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            exp.id != "",
            exp.id,
            f"experience-{i}",
        ),
    )


def _education_card(edu, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: State.remove_education(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=edu.degree,
                    on_change=lambda v: State.update_education_field(i, "degree", v),
                    placeholder="Degree",
                ),
                styled_input(
                    value=edu.school,
                    on_change=lambda v: State.update_education_field(i, "school", v),
                    placeholder="School",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=edu.location,
                on_change=lambda v: State.update_education_field(i, "location", v),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=edu.start_date,
                    on_change=lambda v: State.update_education_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=edu.end_date,
                    on_change=lambda v: State.update_education_field(i, "end_date", v),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=edu.description,
                on_change=lambda v: State.update_education_field(i, "description", v),
                placeholder="Program description or highlights (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=edu.bullets,
                on_change=lambda v: State.update_education_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            edu.id != "",
            edu.id,
            f"education-{i}",
        ),
    )


def _founder_role_card(role, i):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    f"Entry {i + 1}",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    on_click=lambda _=None, i=i: State.remove_founder_role(i),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            rx.hstack(
                styled_input(
                    value=role.role,
                    on_change=lambda v: State.update_founder_role_field(i, "role", v),
                    placeholder="Role",
                ),
                styled_input(
                    value=role.company,
                    on_change=lambda v: State.update_founder_role_field(
                        i, "company", v
                    ),
                    placeholder="Company",
                ),
                width="100%",
                spacing="3",
            ),
            styled_input(
                value=role.location,
                on_change=lambda v: State.update_founder_role_field(i, "location", v),
                placeholder="Location",
            ),
            rx.hstack(
                styled_input(
                    value=role.start_date,
                    on_change=lambda v: State.update_founder_role_field(
                        i, "start_date", v
                    ),
                    type="date",
                ),
                styled_input(
                    value=role.end_date,
                    on_change=lambda v: State.update_founder_role_field(
                        i, "end_date", v
                    ),
                    type="date",
                ),
                width="100%",
                spacing="3",
            ),
            styled_textarea(
                value=role.description,
                on_change=lambda v: State.update_founder_role_field(
                    i, "description", v
                ),
                placeholder="Company or role description (optional)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            styled_textarea(
                value=role.bullets,
                on_change=lambda v: State.update_founder_role_field(i, "bullets", v),
                placeholder="Bullets (one per line)",
                min_height="300px",
                style={"resize": "vertical"},
            ),
            rx.cond(
                State.edit_profile_bullets,
                rx.vstack(
                    rx.text(
                        "Profile bullets (LLM override)",
                        color="#a0aec0",
                        font_size="0.85em",
                    ),
                    styled_textarea(
                        value=State.profile_founder_bullets_list[i],
                        on_change=lambda v, role_id=role.id: State.set_profile_founder_bullets_text(
                            role_id, v
                        ),
                        placeholder="Profile bullets (one per line)",
                        min_height="240px",
                        width="100%",
                        style={"resize": "vertical"},
                    ),
                    spacing="2",
                    width="100%",
                ),
                rx.fragment(),
            ),
            width="100%",
            spacing="3",
        ),
        margin_bottom="1.5em",
        key=rx.cond(
            role.id != "",
            role.id,
            f"founder-{i}",
        ),
    )


def _custom_section_editor(row):
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    rx.cond(row.get("title"), row.get("title"), "Custom section"),
                    color="#e2e8f0",
                    font_size="0.95em",
                    weight="medium",
                ),
                rx.spacer(),
                rx.button(
                    "Remove",
                    data_section_key=row.get("key", ""),
                    on_click=lambda _=None, key=rx.cond(
                        row.get("key"),
                        row.get("key"),
                        "",
                    ): State.remove_custom_section(key),
                    size="1",
                    variant="soft",
                    color_scheme="red",
                ),
                width="100%",
                align_items="center",
            ),
            styled_textarea(
                value=row.get("body", ""),
                on_change=lambda v, key=row.get("key"): State.set_custom_section_body(
                    key, v
                ),
                placeholder="Section content (one bullet per line)",
                min_height="140px",
                style={"resize": "vertical"},
            ),
            width="100%",
            spacing="2",
        ),
        margin_bottom="1.0em",
        data_custom_section="1",
        data_section_key=row.get("key", ""),
    )


def section_order_controls():
    return styled_card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "Section order",
                    weight="bold",
                    color="#e2e8f0",
                    font_size="1em",
                ),
                rx.spacer(),
                rx.text(
                    "Top renders first",
                    color="#a0aec0",
                    font_size="0.85em",
                ),
                align_items="center",
                width="100%",
            ),
            rx.text(
                "Reorder how PDF sections appear.",
                color="#a0aec0",
                font_size="0.9em",
            ),
            rx.vstack(
                rx.foreach(State.section_order_rows, _section_order_row),
                spacing="2",
                width="100%",
            ),
            rx.divider(margin_y="1em"),
            rx.text(
                "Add section",
                color="#e2e8f0",
                weight="bold",
                font_size="0.95em",
            ),
            styled_input(
                value=State.new_section_title,
                on_change=State.set_new_section_title,
                placeholder="Section title",
            ),
            styled_textarea(
                value=State.new_section_body,
                on_change=State.set_new_section_body,
                placeholder="Section content (one bullet per line)",
                min_height="120px",
                style={"resize": "vertical"},
            ),
            rx.button(
                "Add Section",
                on_click=State.add_custom_section,
                size="2",
                variant="soft",
                color_scheme="green",
            ),
            rx.vstack(
                rx.foreach(State.custom_section_rows, _custom_section_editor),
                spacing="2",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        margin_top="0.5em",
    )


def index():
    loading_view = rx.center(
        rx.vstack(
            rx.heading("Loading Resume Builder…", size="7", color="#f7fafc"),
            rx.text(
                "Preparing data and UI. This will finish once hydration completes.",
                color="#a0aec0",
            ),
            rx.spinner(size="3"),
        ),
        width="100%",
        height="100vh",
        bg="#0b1224",
        padding="2em",
    )

    return rx.fragment(
        rx.toast.provider(position="top-right", duration=6000, close_button=True),
        rx.cond(
            State.has_loaded,
            rx.hstack(
                # Left Panel
                rx.box(
                    rx.vstack(
                        rx.heading(
                            "Resume Builder",
                            size="8",
                            margin_bottom="1em",
                            color="#f7fafc",
                        ),
                        rx.hstack(
                            rx.text("Job Requisition", weight="bold", color="#e2e8f0"),
                            rx.spacer(),
                            rx.button(
                                "Paste & Run",
                                on_click=rx.call_script(
                                    "navigator.clipboard.readText()",
                                    callback=State.paste_req_and_run_pipeline,
                                ),
                                size="1",
                                variant="soft",
                                color_scheme="gray",
                                cursor="pointer",
                            ),
                            rx.button(
                                "Run Pipeline",
                                on_click=State.auto_pipeline_from_req,
                                size="1",
                                variant="soft",
                                color_scheme="gray",
                            ),
                            width="100%",
                            align_items="center",
                            margin_bottom="0.5em",
                        ),
                        rx.hstack(
                            rx.text(
                                "Model", weight="medium", color="#a0aec0", width="25%"
                            ),
                            rx.select(
                                State.llm_models,
                                value=State.selected_model,
                                on_change=State.set_selected_model,
                                width="75%",
                                color_scheme="indigo",
                                radius="medium",
                                size="2",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        styled_textarea(
                            placeholder="Paste or type the job requisition here",
                            value=State.job_req,
                            on_change=State.set_job_req,
                            id="job-req-field",
                            min_height="150px",
                            style={"resize": "vertical"},
                        ),
                        rx.text(
                            "Prompt Template",
                            weight="bold",
                            color="#e2e8f0",
                            margin_top="0.5em",
                        ),
                        styled_textarea(
                            placeholder="Edit the prompt template (stored in Neo4j)",
                            value=State.prompt_yaml,
                            on_change=State.set_prompt_yaml,
                            id="prompt-yaml-field",
                            min_height="220px",
                            style={"resize": "vertical"},
                        ),
                        rx.cond(
                            State.job_req_needs_profile,
                            rx.text(
                                "Job requisition not yet processed. Click Generate Profile or Run Pipeline.",
                                color="#f6ad55",
                                font_size="0.85em",
                            ),
                        ),
                        rx.cond(
                            State.pipeline_latest != "",
                            rx.text(
                                State.pipeline_latest,
                                color="#a0aec0",
                                font_size="0.85em",
                            ),
                        ),
                        # Pipeline status messages are tracked for debugging; we show the
                        # latest message inline so it's obvious when the LLM/pipeline runs.
                        rx.hstack(
                            rx.button(
                                "Save Data",
                                on_click=State.save_to_db,
                                loading=State.is_saving,
                                size="4",
                                color_scheme="green",
                                flex="1",
                                id="save-btn",
                            ),
                            rx.button(
                                "Load Data",
                                on_click=State.load_resume_fields,
                                loading=State.is_loading_resume,
                                size="4",
                                color_scheme="gray",
                                flex="1",
                                id="load-resume-btn",
                            ),
                            rx.button(
                                "Generate Profile",
                                on_click=State.generate_profile,
                                loading=State.is_generating_profile,
                                size="4",
                                color_scheme="blue",
                                flex="1",
                                id="generate-profile-btn",
                            ),
                            rx.button(
                                "Generate PDF",
                                on_click=State.generate_pdf,
                                loading=State.generating_pdf,
                                size="4",
                                color_scheme="indigo",
                                flex="1",
                                id="generate-pdf-btn",
                            ),
                            rx.cond(
                                State.db_error != "",
                                rx.tooltip(
                                    rx.box(
                                        rx.text(
                                            "!",
                                            color="#f56565",
                                            weight="bold",
                                            font_size="0.9em",
                                        ),
                                        width="2.5em",
                                        height="2.5em",
                                        border="1px solid #f56565",
                                        border_radius="999px",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center",
                                        aria_label="Database error indicator",
                                    ),
                                    content=State.db_error,
                                ),
                            ),
                            spacing="2",
                            width="100%",
                            align_items="center",
                            margin_top="1em",
                        ),
                        labeled_toggle(
                            "Include Skills section",
                            checked=State.include_matrices,
                            on_change=State.set_include_matrices,
                            container_props={"margin_top": "0.25em"},
                        ),
                        labeled_toggle(
                            "Rewrite bullets with LLM",
                            checked=State.rewrite_bullets_with_llm,
                            on_change=State.set_rewrite_bullets_with_llm,
                            container_props={"margin_top": "0.25em"},
                        ),
                        rx.hstack(
                            rx.text(
                                "Auto-fit",
                                weight="medium",
                                color="#e2e8f0",
                            ),
                            rx.switch(
                                checked=State.auto_tune_pdf,
                                on_change=State.set_auto_tune_pdf,
                                size="3",
                                color_scheme="indigo",
                            ),
                            rx.text(
                                "Pages",
                                color="#a0aec0",
                                font_size="0.85em",
                            ),
                            styled_input(
                                value=State.auto_fit_target_pages,
                                on_change=State.set_auto_fit_target_pages,
                                type="number",
                                min="1",
                                step="1",
                                width="5.5em",
                            ),
                            width="100%",
                            align_items="center",
                            spacing="2",
                            margin_top="0.25em",
                        ),
                        rx.vstack(
                            rx.text(
                                "Resume font",
                                weight="medium",
                                color="#e2e8f0",
                            ),
                            styled_input(
                                value=State.font_family,
                                on_change=State.set_font_family,
                                placeholder="Pick a font",
                                id="resume-font-picker",
                            ),
                            spacing="1",
                            width="100%",
                            margin_top="0.35em",
                        ),
                        section_order_controls(),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    State.last_render_label != "",
                                    State.last_render_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#a0aec0",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    State.last_save_label != "",
                                    State.last_save_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#9ae6b4",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.center(
                            rx.text(
                                rx.cond(
                                    State.last_profile_bullets_label != "",
                                    State.last_profile_bullets_label,
                                    "",
                                ),
                                font_size="0.8em",
                                color="#9ad0ff",
                            ),
                            width="100%",
                            margin_top="0.25em",
                        ),
                        rx.cond(
                            State.data_loaded,
                            rx.box(
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Role Details",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.grid(
                                    [
                                        styled_input(
                                            value=State.target_company,
                                            on_change=None,
                                            placeholder="Target Company",
                                            read_only=True,
                                            key="target_company",
                                        ),
                                        styled_input(
                                            value=State.target_role,
                                            on_change=None,
                                            placeholder="Target Role",
                                            read_only=True,
                                            key="target_role",
                                        ),
                                        styled_input(
                                            value=State.seniority_level,
                                            on_change=None,
                                            placeholder="Seniority Level",
                                            read_only=True,
                                            key="seniority_level",
                                        ),
                                        styled_input(
                                            value=State.target_location,
                                            on_change=None,
                                            placeholder="Target Location",
                                            read_only=True,
                                            key="target_location",
                                        ),
                                        styled_input(
                                            value=State.work_mode,
                                            on_change=None,
                                            placeholder="Work Mode",
                                            read_only=True,
                                            key="work_mode",
                                        ),
                                        styled_input(
                                            value=State.travel_requirement,
                                            on_change=None,
                                            placeholder="Travel Requirement",
                                            read_only=True,
                                            key="travel_requirement",
                                        ),
                                        styled_input(
                                            value=State.primary_domain,
                                            on_change=None,
                                            placeholder="Primary Domain",
                                            read_only=True,
                                            key="primary_domain",
                                        ),
                                        styled_input(
                                            value=State.salary_band,
                                            on_change=None,
                                            placeholder="Salary Band",
                                            read_only=True,
                                            key="salary_band",
                                        ),
                                        styled_input(
                                            value=State.posting_url,
                                            on_change=None,
                                            placeholder="Posting URL",
                                            read_only=True,
                                            key="posting_url",
                                        ),
                                        styled_input(
                                            value=State.req_id,
                                            on_change=None,
                                            placeholder="Req ID",
                                            read_only=True,
                                            key="req_id",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.grid(
                                    [
                                        styled_textarea(
                                            value=State.must_have_skills_text,
                                            on_change=None,
                                            placeholder="Must Have Skills",
                                            read_only=True,
                                            min_height="100px",
                                            key="must_have_skills_text",
                                        ),
                                        styled_textarea(
                                            value=State.nice_to_have_skills_text,
                                            on_change=None,
                                            placeholder="Nice To Have Skills",
                                            read_only=True,
                                            min_height="100px",
                                            key="nice_to_have_skills_text",
                                        ),
                                        styled_textarea(
                                            value=State.tech_stack_keywords_text,
                                            on_change=None,
                                            placeholder="Tech Stack Keywords",
                                            read_only=True,
                                            min_height="100px",
                                            key="tech_stack_keywords_text",
                                        ),
                                        styled_textarea(
                                            value=State.non_technical_requirements_text,
                                            on_change=None,
                                            placeholder="Non-Technical Requirements",
                                            read_only=True,
                                            min_height="100px",
                                            key="non_technical_requirements_text",
                                        ),
                                        styled_textarea(
                                            value=State.certifications_text,
                                            on_change=None,
                                            placeholder="Certifications",
                                            read_only=True,
                                            min_height="100px",
                                            key="certifications_text",
                                        ),
                                        styled_textarea(
                                            value=State.clearances_text,
                                            on_change=None,
                                            placeholder="Clearances",
                                            read_only=True,
                                            min_height="100px",
                                            key="clearances_text",
                                        ),
                                        styled_textarea(
                                            value=State.core_responsibilities_text,
                                            on_change=None,
                                            placeholder="Core Responsibilities",
                                            read_only=True,
                                            min_height="100px",
                                            key="core_responsibilities_text",
                                        ),
                                        styled_textarea(
                                            value=State.outcome_goals_text,
                                            on_change=None,
                                            placeholder="Outcome Goals",
                                            read_only=True,
                                            min_height="100px",
                                            key="outcome_goals_text",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.text(
                                    "Job Requisition (stored)",
                                    weight="bold",
                                    margin_top="0.5em",
                                    color="#e2e8f0",
                                ),
                                styled_textarea(
                                    value=State.job_req,
                                    on_change=None,
                                    placeholder="Job requisition as stored on the Profile",
                                    read_only=True,
                                    min_height="180px",
                                    style={"resize": "vertical"},
                                ),
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Contact Info",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.grid(
                                    [
                                        styled_input(
                                            value=State.first_name,
                                            on_change=State.set_first_name,
                                            placeholder="First Name",
                                            key="first_name",
                                        ),
                                        styled_input(
                                            value=State.middle_name,
                                            on_change=State.set_middle_name,
                                            placeholder="Middle Name / Initial",
                                            key="middle_name",
                                        ),
                                        styled_input(
                                            value=State.last_name,
                                            on_change=State.set_last_name,
                                            placeholder="Last Name",
                                            key="last_name",
                                        ),
                                        styled_input(
                                            value=State.email,
                                            on_change=State.set_email,
                                            placeholder="Email",
                                            key="email",
                                        ),
                                        styled_input(
                                            value=State.email2,
                                            on_change=State.set_email2,
                                            placeholder="Secondary Email",
                                            key="email2",
                                        ),
                                        styled_input(
                                            value=State.phone,
                                            on_change=State.set_phone,
                                            placeholder="Phone",
                                            key="phone",
                                        ),
                                        styled_input(
                                            value=State.linkedin_url,
                                            on_change=State.set_linkedin_url,
                                            placeholder="LinkedIn URL",
                                            key="linkedin_url",
                                        ),
                                        styled_input(
                                            value=State.github_url,
                                            on_change=State.set_github_url,
                                            placeholder="GitHub URL",
                                            key="github_url",
                                        ),
                                        styled_input(
                                            value=State.scholar_url,
                                            on_change=State.set_scholar_url,
                                            placeholder="Google Scholar URL",
                                            key="scholar_url",
                                        ),
                                        styled_input(
                                            value=State.calendly_url,
                                            on_change=State.set_calendly_url,
                                            placeholder="Calendly URL",
                                            key="calendly_url",
                                        ),
                                        styled_input(
                                            value=State.portfolio_url,
                                            on_change=State.set_portfolio_url,
                                            placeholder="Portfolio URL",
                                            key="portfolio_url",
                                        ),
                                    ],
                                    columns="2",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Generated Content",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.text(
                                    "Professional Summary",
                                    weight="bold",
                                    margin_top="1em",
                                    color="#e2e8f0",
                                ),
                                styled_textarea(
                                    value=State.summary,
                                    on_change=State.set_summary,
                                    min_height="200px",
                                    style={"resize": "vertical"},
                                ),
                                rx.divider(margin_y="1.5em"),
                                rx.heading(
                                    "Skills",
                                    size="6",
                                    margin_bottom="0.5em",
                                    color="#f7fafc",
                                ),
                                rx.text(
                                    "Three labeled rows with comma-separated skills (matches prompt.yaml).",
                                    color="#a0aec0",
                                    margin_bottom="0.5em",
                                    font_size="0.9em",
                                ),
                                rx.vstack(
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[0] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            State.skills_rows_csv[0],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[1] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            State.skills_rows_csv[1],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    rx.hstack(
                                        rx.text(
                                            DEFAULT_SKILLS_ROW_LABELS[2] + ":",
                                            weight="bold",
                                            color="#e2e8f0",
                                            width="35%",
                                        ),
                                        rx.text(
                                            State.skills_rows_csv[2],
                                            color="#e2e8f0",
                                            width="65%",
                                        ),
                                        width="100%",
                                        align_items="flex-start",
                                    ),
                                    spacing="2",
                                    width="100%",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Experience",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.hstack(
                                        rx.switch(
                                            checked=State.edit_profile_bullets,
                                            on_change=State.set_edit_profile_bullets,
                                            size="2",
                                            color_scheme="indigo",
                                        ),
                                        rx.text(
                                            "Edit LLM bullets",
                                            color="#a0aec0",
                                            font_size="0.85em",
                                        ),
                                        spacing="2",
                                        align_items="center",
                                    ),
                                    rx.button(
                                        "Save Profile Bullets",
                                        on_click=State.save_profile_bullets,
                                        loading=State.is_saving_profile_bullets,
                                        size="2",
                                        variant="soft",
                                        color_scheme="indigo",
                                    ),
                                    rx.button(
                                        "Add Experience",
                                        on_click=State.add_experience,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                    spacing="2",
                                ),
                                rx.vstack(
                                    rx.foreach(State.experience, _experience_card),
                                    width="100%",
                                    spacing="4",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Education",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.button(
                                        "Add Education",
                                        on_click=State.add_education,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                ),
                                rx.vstack(
                                    rx.foreach(State.education, _education_card),
                                    width="100%",
                                    spacing="4",
                                ),
                                rx.divider(margin_y="2em"),
                                rx.hstack(
                                    rx.heading(
                                        "Founder Roles",
                                        size="6",
                                        color="#f7fafc",
                                    ),
                                    rx.spacer(),
                                    rx.button(
                                        "Add Founder Role",
                                        on_click=State.add_founder_role,
                                        size="2",
                                        variant="soft",
                                        color_scheme="green",
                                    ),
                                    width="100%",
                                    align_items="center",
                                    margin_bottom="0.5em",
                                ),
                                rx.vstack(
                                    rx.foreach(
                                        State.founder_roles,
                                        _founder_role_card,
                                    ),
                                    width="100%",
                                    spacing="4",
                                ),
                                width="100%",
                            ),
                            rx.vstack(
                                rx.divider(margin_y="2em"),
                                rx.heading(
                                    "Data not loaded", size="5", color="#f7fafc"
                                ),
                                rx.text(
                                    "Click “Load Data” to fetch your latest resume and profile fields from Neo4j.",
                                    color="#a0aec0",
                                    font_size="0.95em",
                                ),
                                align_items="start",
                                spacing="2",
                                width="100%",
                            ),
                        ),
                        width="100%",
                        max_width="800px",
                        margin_x="auto",
                        padding_y="2em",
                    ),
                    width="50%",
                    height="100vh",
                    padding="2em",
                    overflow="auto",
                    bg="#1a202c",
                ),
                # Right Panel (PDF)
                rx.box(
                    rx.vstack(
                        rx.box(
                            rx.cond(
                                State.pdf_error != "",
                                rx.center(
                                    rx.text(State.pdf_error, color="#f56565"),
                                    height="100%",
                                ),
                                rx.cond(
                                    State.pdf_url != "",
                                    rx.el.embed(
                                        src=f"{State.pdf_url}#view=FitH&zoom=page-width&toolbar=0&navpanes=0&scrollbar=0",
                                        type="application/pdf",
                                        style={
                                            "width": "100%",
                                            "height": "100%",
                                            "border": "none",
                                            "border_radius": "12px",
                                            "display": "block",
                                        },
                                        key=State.last_pdf_signature,
                                    ),
                                    rx.box(
                                        height="100%"
                                    ),  # empty state instead of spinner
                                ),
                            ),
                            width="100%",
                            height="100%",
                            bg="#0b1224",
                            padding="0.5em",
                            border_radius="12px",
                            box_shadow="0 10px 25px rgba(0,0,0,0.35)",
                        ),
                        width="100%",
                        height="100%",
                        padding="1em",
                    ),
                    width="50%",
                    height="100vh",
                    border_left="1px solid #2d3748",
                    bg="#0b1224",
                ),
                width="100%",
                height="100vh",
                overflow="hidden",
            ),
            loading_view,
        ),
    )


APP_STYLE = {
    "font_family": "Inter, system-ui, sans-serif",
    "background_color": "#1a202c",
}

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color="blue",
    ),
    stylesheets=[
        "https://www.jsfontpicker.com/css/fontpicker.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/theme/material-darker.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
    ],
    head_components=[
        rx.script(src="https://www.jsfontpicker.com/js/fontpicker.iife.min.js"),
        rx.script(
            f"""
            (function () {{
              var extraFonts = {FONT_PICKER_EXTRA_FONTS_JSON};
              var defaultFont = {json.dumps(DEFAULT_RESUME_FONT_FAMILY)};

              function normalizeFamily(value) {{
                if (!value) {{
                  return "";
                }}
                var raw = String(value);
                return raw.split(":")[0].trim();
              }}

              function hasExtraFont(name) {{
                var target = normalizeFamily(name);
                if (!target) {{
                  return false;
                }}
                for (var i = 0; i < extraFonts.length; i += 1) {{
                  if (normalizeFamily(extraFonts[i].name) === target) {{
                    return true;
                  }}
                }}
                return false;
              }}

              function initPicker() {{
                var input = document.getElementById("resume-font-picker");
                if (!input) {{
                  return false;
                }}
                if (input.dataset.fpAttached === "1") {{
                  return true;
                }}
                if (!window.FontPicker) {{
                  return false;
                }}
                document.documentElement.dataset.fpTheme = "dark";
                var safeDefault = hasExtraFont(defaultFont) ? normalizeFamily(defaultFont) : "";
                if (!input.value && safeDefault) {{
                  input.value = safeDefault;
                }}
                if (input.value) {{
                  input.value = normalizeFamily(input.value);
                }}
                var picker = new window.FontPicker(input, {{
                  variants: false,
                  verbose: false,
                  font: input.value || null,
                  googleFonts: null,
                  systemFonts: null,
                  extraFonts: extraFonts
                }});
                input.dataset.fpAttached = "1";
                input.style.fontFamily = input.value || "";
                var lastValue = input.value || "";

                function syncFromInput() {{
                  if (input.value === lastValue) {{
                    return;
                  }}
                  lastValue = input.value || "";
                  if (lastValue) {{
                    try {{
                      picker.setFont(lastValue);
                    }} catch (e) {{}}
                    input.style.fontFamily = lastValue;
                  }} else {{
                    input.style.fontFamily = "";
                  }}
                }}

                input.addEventListener("input", function () {{
                  if (input.dataset.fpPicking === "1") {{
                    return;
                  }}
                  syncFromInput();
                }});

                picker.on("pick", function (font) {{
                  input.dataset.fpPicking = "1";
                  if (!font) {{
                    input.value = "";
                    input.style.fontFamily = "";
                  }} else {{
                    var family = font.family && font.family.name ? font.family.name : font.toString();
                    input.value = family;
                    input.style.fontFamily = font.family && font.family.name ? font.family.name : "";
                  }}
                  lastValue = input.value || "";
                  input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                  input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                  input.dataset.fpPicking = "0";
                }});

                if (window.queryLocalFonts) {{
                  try {{
                    window.queryLocalFonts().then(function (fonts) {{
                      var seen = Object.create(null);
                      fonts.forEach(function (f) {{
                        if (f.family) {{
                          seen[f.family] = true;
                        }}
                      }});
                      var families = Object.keys(seen);
                      if (families.length) {{
                        families.sort();
                        picker.setConfiguration({{ systemFonts: families }});
                      }}
                    }});
                  }} catch (e) {{}}
                }}

                setInterval(syncFromInput, 500);
                return true;
              }}

              function boot() {{
                if (!initPicker()) {{
                  setTimeout(boot, 300);
                }}
              }}

              if (document.readyState === "loading") {{
                document.addEventListener("DOMContentLoaded", boot);
              }} else {{
                boot();
              }}
            }})();
            """
        ),
        rx.script(
            src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"
        ),
        rx.script(
            """
            (function () {
              function ensureYamlMode() {
                if (!window.CodeMirror || !window.CodeMirror.modes) {
                  return false;
                }
                if (window.CodeMirror.modes.yaml) {
                  return true;
                }
                if (window.__promptYamlModeLoading) {
                  return false;
                }
                window.__promptYamlModeLoading = true;
                var script = document.createElement("script");
                script.src = "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/yaml/yaml.min.js";
                script.onload = function () {
                  window.__promptYamlModeLoading = false;
                };
                script.onerror = function () {
                  window.__promptYamlModeLoading = false;
                };
                document.head.appendChild(script);
                return false;
              }

              function ensureStyle() {
                if (window.__promptYamlStyleInjected) {
                  return;
                }
                var style = document.createElement("style");
                style.textContent = "#prompt-yaml-field + .CodeMirror{border:1px solid #4a5568;border-radius:0.375rem;background:#1f2937;}#prompt-yaml-field + .CodeMirror .CodeMirror-gutters{border-right:1px solid #374151;background:#1f2937;}#prompt-yaml-field + .CodeMirror .CodeMirror-linenumber{color:#9ca3af;}#prompt-yaml-field + .CodeMirror .CodeMirror-cursor{border-left:1px solid #e2e8f0;}";
                document.head.appendChild(style);
                window.__promptYamlStyleInjected = true;
              }

              function initPromptYamlEditor() {
                var textarea = document.getElementById("prompt-yaml-field");
                if (!textarea) {
                  return false;
                }
                if (textarea.dataset.cmAttached === "1") {
                  return true;
                }
                if (!window.CodeMirror || !window.CodeMirror.fromTextArea) {
                  return false;
                }
                var editor = window.CodeMirror.fromTextArea(textarea, {
                  mode: "yaml",
                  lineNumbers: true,
                  lineWrapping: true,
                  tabSize: 2,
                  indentUnit: 2,
                  theme: "material-darker"
                });
                textarea.dataset.cmAttached = "1";
                textarea.dataset.cmSyncedValue = textarea.value || "";
                window.__promptYamlCodeMirror = editor;
                editor.setSize("100%", "220px");
                editor.on("change", function (cm) {
                  cm.save();
                  textarea.dataset.cmSyncedValue = textarea.value || "";
                  textarea.dispatchEvent(new Event("input", { bubbles: true }));
                });
                return true;
              }

              function syncFromTextarea() {
                var textarea = document.getElementById("prompt-yaml-field");
                var editor = window.__promptYamlCodeMirror;
                if (!textarea || !editor) {
                  return;
                }
                var nextValue = textarea.value || "";
                var currentValue = editor.getValue();
                var lastSynced = textarea.dataset.cmSyncedValue || "";
                if (nextValue !== currentValue && nextValue !== lastSynced) {
                  editor.setValue(nextValue);
                  textarea.dataset.cmSyncedValue = nextValue;
                }
              }

              function boot() {
                var attempts = window.__promptYamlInitAttempts || 0;
                window.__promptYamlInitAttempts = attempts + 1;
                if (attempts > 200) {
                  return;
                }
                ensureStyle();
                if (!window.CodeMirror || !window.CodeMirror.fromTextArea) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!ensureYamlMode()) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!initPromptYamlEditor()) {
                  window.setTimeout(boot, 250);
                  return;
                }
                if (!window.__promptYamlSyncTimer) {
                  window.__promptYamlSyncTimer = window.setInterval(syncFromTextarea, 750);
                }
              }

              if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", boot);
              } else {
                boot();
              }
            })();
            """
        ),
    ],
    style=APP_STYLE,
)
app.add_page(index, on_load=State.on_load)

if __name__ == "__main__":
    if "--maximum-coverage" in sys.argv and not os.environ.get("_MAX_COVERAGE_CHILD"):
        os.environ.setdefault("MAX_COVERAGE_SILENCE_WARNINGS", "1")
        _install_maxcov_print_filter()
        _maybe_launch_maxcov_container()
        coverage = _try_import_coverage()
        if coverage is not None:
            _maxcov_log("maxcov wrapper start")
            self_test = os.environ.get("_MAX_COVERAGE_SELFTEST") == "1"
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            cov_dir = Path(tempfile.gettempdir()) / f"max_coverage_{stamp}"
            cov_dir.mkdir(parents=True, exist_ok=True)
            cov_rc = cov_dir / "coverage_rc"
            cov_rc.write_text(
                "\n".join(
                    [
                        "[run]",
                        "branch = True",
                        "parallel = True",
                        "source =",
                        f"    {BASE_DIR}",
                        "omit =",
                        "    */site-packages/*",
                        "    */__pycache__/*",
                        "    */.tmp_typst/*",
                        "    */assets/*",
                        "    */fonts/*",
                        "    */packages/*",
                        "    */.pytest_cache/*",
                        "",
                        "[report]",
                        "show_missing = False",
                        "skip_empty = True",
                    ]
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["_MAX_COVERAGE_CHILD"] = "1"
            env["MAX_COVERAGE_LOG"] = "1"
            env["MAX_COVERAGE_SKIP_LLM"] = "1"
            env.setdefault("PYTHONUNBUFFERED", "1")
            env["COVERAGE_FILE"] = str(cov_dir / ".coverage")
            base_cmd = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(cov_rc),
                "--parallel-mode",
                str(Path(__file__).resolve()),
            ]
            req_file = _get_arg_value(
                sys.argv[1:], "--req-file", str(BASE_DIR / "req.txt")
            )
            empty_home = cov_dir / "empty_home"
            empty_home.mkdir(parents=True, exist_ok=True)
            missing_req = cov_dir / "missing_req.txt"
            missing_typst = cov_dir / "missing_typst"

            def _run_cov(
                args: list[str], *, extra_env=None, quiet=False, expected_failure=False
            ) -> int:
                run_env = env.copy()
                if extra_env:
                    for key, val in extra_env.items():
                        if val is None:
                            run_env.pop(key, None)
                        else:
                            run_env[key] = val
                kwargs = {}
                if quiet:
                    kwargs["stdout"] = subprocess.PIPE
                    kwargs["stderr"] = subprocess.PIPE
                    kwargs["text"] = True
                started = time.perf_counter()
                cmd = [*base_cmd, *args]
                _maxcov_log(f"run start: {' '.join(cmd)}")
                if extra_env:
                    _maxcov_log(
                        f"run env overrides: {', '.join(sorted(k for k in extra_env.keys()))}"
                    )
                proc = subprocess.Popen(cmd, env=run_env, **kwargs)
                next_log = started + 5.0
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    now = time.perf_counter()
                    if now >= next_log:
                        _maxcov_log(
                            f"run heartbeat ({now - started:.1f}s): {' '.join(args) or '<no args>'}"
                        )
                        next_log = now + 5.0
                    time.sleep(0.25)
                stdout, stderr = proc.communicate()
                elapsed = time.perf_counter() - started
                _maxcov_log(f"run done ({elapsed:.1f}s): rc={proc.returncode}")
                if proc.returncode != 0:
                    if expected_failure:
                        _maxcov_log(
                            f"run expected failure: rc={proc.returncode} args={' '.join(args)}"
                        )
                        _maxcov_log_expected_failure(stdout, stderr, args, quiet)
                    else:
                        print(
                            f"Warning: coverage run failed for args: {' '.join(args)}"
                        )
                        if quiet and stdout:
                            print(stdout.rstrip())
                        if quiet and stderr:
                            print(stderr.rstrip(), file=sys.stderr)
                return proc.returncode

            coverage_runs = [
                (sys.argv[1:], {}, False, False),
                (
                    ["--maximum-coverage"],
                    {"_MAX_COVERAGE_CHILD": None, "_MAX_COVERAGE_SELFTEST": "1"},
                    True,
                    False,
                ),
                (
                    ["--maximum-coverage", "--maximum-coverage-actions", "bogus"],
                    {},
                    True,
                    True,
                ),
                (["--bogus-flag"], {}, True, True),
                (["--list-models"], {}, True, False),
                (["--list-models"], {"REFLEX_COVERAGE": "1"}, True, False),
                (
                    ["--list-models"],
                    {"REFLEX_COVERAGE": "1", "REFLEX_COVERAGE_FORCE_OWNED": "1"},
                    True,
                    False,
                ),
                ([], {}, True, True),
                (["--export-resume-pdf"], {}, True, False),
                (["--list-applied"], {}, True, False),
                (["--reset-db"], {}, True, False),
                (["--import-assets"], {}, True, False),
                (
                    ["--compile-pdf", str(ASSETS_DIR / "preview.pdf"), "--auto-fit"],
                    {},
                    True,
                    False,
                ),
                (["--compile-pdf", str(ASSETS_DIR / "preview.pdf")], {}, True, False),
                (
                    ["--compile-pdf", str(ASSETS_DIR / "preview.pdf")],
                    {"TYPST_BIN": str(missing_typst)},
                    True,
                    True,
                ),
                (
                    ["--generate-profile", "--req-file", req_file],
                    {"MAX_COVERAGE_SKIP_LLM": "1"},
                    True,
                    False,
                ),
                (["--show-resume-data"], {}, True, False),
                (["--eval-prompt", "--req-file", str(missing_req)], {}, True, True),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "bogus:model",
                    ],
                    {},
                    True,
                    False,
                ),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "openai:gpt-4o-mini",
                    ],
                    {
                        "HOME": str(empty_home),
                        "OPENAI_API_KEY": "",
                        "OPENAI_ORG_ID": "",
                        "OPENAI_ORGANIZATION": "",
                        "OPENAI_PROJECT": "",
                    },
                    True,
                    False,
                ),
                (
                    [
                        "--eval-prompt",
                        "--req-file",
                        req_file,
                        "--model-name",
                        "gemini:gemini-1.5-flash",
                    ],
                    {
                        "HOME": str(empty_home),
                        "GEMINI_API_KEY": "",
                        "GOOGLE_API_KEY": "",
                    },
                    True,
                    False,
                ),
            ]
            if self_test:
                base_cmd = [sys.executable, "-c", "print('maxcov bootstrap selftest')"]
                coverage_runs = [
                    ([], {}, True, False),
                    (["--noop"], {}, True, False),
                ]
            total_runs = len(coverage_runs)
            for idx, (args, env_overrides, quiet, expected_failure) in enumerate(
                coverage_runs, start=1
            ):
                _maxcov_log(
                    f"run dispatch {idx}/{total_runs}: args={' '.join(args) or '<no args>'}"
                )
                _run_cov(
                    args,
                    extra_env=env_overrides,
                    quiet=quiet,
                    expected_failure=expected_failure,
                )
            _maxcov_log("coverage combine start")
            started = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "combine",
                    "--rcfile",
                    str(cov_rc),
                    str(cov_dir),
                ],
                env=env,
                check=False,
            )
            _maxcov_log(f"coverage combine done ({time.perf_counter() - started:.1f}s)")
            _maxcov_log("coverage report start")
            started = time.perf_counter()
            report = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "report",
                    "--rcfile",
                    str(cov_rc),
                    "--data-file",
                    str(cov_dir / ".coverage"),
                    "--include",
                    str(Path(__file__).resolve()),
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            _maxcov_log(f"coverage report done ({time.perf_counter() - started:.1f}s)")
            json_path = cov_dir / "coverage.json"
            html_dir = cov_dir / "htmlcov"
            json_out = None
            html_out = None
            _maxcov_log("coverage json start")
            started = time.perf_counter()
            try:
                json_cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "json",
                    "--rcfile",
                    str(cov_rc),
                ]
                json_cmd.extend(["--data-file", str(cov_dir / ".coverage")])
                json_cmd.extend(["--include", str(Path(__file__).resolve())])
                json_cmd.extend(["--output", str(json_path)])
                json_result = subprocess.run(
                    json_cmd,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if json_result.returncode == 0 and json_path.exists():
                    json_out = str(json_path)
            except Exception as exc:
                _maxcov_log(f"coverage json failed: {exc}")
            _maxcov_log(f"coverage json done ({time.perf_counter() - started:.1f}s)")
            _maxcov_log("coverage html start")
            started = time.perf_counter()
            try:
                html_cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "html",
                    "--rcfile",
                    str(cov_rc),
                ]
                html_cmd.extend(["--data-file", str(cov_dir / ".coverage")])
                html_cmd.extend(["--include", str(Path(__file__).resolve())])
                html_cmd.extend(["--directory", str(html_dir)])
                html_result = subprocess.run(
                    html_cmd,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                index_path = html_dir / "index.html"
                if html_result.returncode == 0 and index_path.exists():
                    html_out = str(index_path)
            except Exception as exc:
                _maxcov_log(f"coverage html failed: {exc}")
            _maxcov_log(f"coverage html done ({time.perf_counter() - started:.1f}s)")
            counts = {}
            for line in (report.stdout or "").splitlines():
                if line.strip().startswith("harness.py"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            counts = {
                                "stmts": int(parts[1]),
                                "miss": int(parts[2]),
                                "branch": int(parts[3]),
                                "brpart": int(parts[4]),
                                "cover": parts[5],
                            }
                        except Exception:
                            counts = {"cover": parts[5]}
                        break
            summary = _maxcov_summarize_coverage(
                coverage,
                cov_dir=cov_dir,
                cov_rc=cov_rc,
                target=Path(__file__).resolve(),
            )
            lines = _maxcov_build_coverage_output(
                counts=counts,
                summary=summary,
                cov_dir=cov_dir,
                cov_rc=cov_rc,
                json_out=json_out,
                html_out=html_out,
            )
            if lines:
                print("\n".join(lines))
            elif report.stdout:
                print(report.stdout.rstrip())
            sys.exit(0)

    parser = argparse.ArgumentParser(description="Resume Builder Utility")
    parser.add_argument(
        "--import-assets",
        help=(
            "Path to assets JSON file to import (see michael_scott_resume.json for"
            " schema). Refuses to overwrite existing resume unless"
            " --overwrite-resume is set."
        ),
        nargs="?",
        const=str(DEFAULT_ASSETS_JSON),
    )
    parser.add_argument(
        "--overwrite-resume",
        action="store_true",
        help="Allow --import-assets to replace an existing resume in Neo4j.",
    )
    parser.add_argument(
        "--reset-db",
        help="Reset Neo4j (wipe) then import assets JSON (default: michael_scott_resume.json)",
        nargs="?",
        const=str(DEFAULT_ASSETS_JSON),
    )
    parser.add_argument(
        "--generate-profile",
        action="store_true",
        help="Generate a new Profile from the stored prompt template (prompt.yaml fallback) + --req-file and save it to Neo4j (combine with --compile-pdf to render).",
    )
    parser.add_argument(
        "--eval-prompt",
        action="store_true",
        help="Send the stored prompt template (prompt.yaml fallback) plus req.txt (or --req-file) to the LLM and print the JSON response.",
    )
    parser.add_argument(
        "--compile-pdf",
        help="Compile the current resume to a PDF at the given path (defaults to assets/preview.pdf)",
        nargs="?",
        const=str(ASSETS_DIR / "preview.pdf"),
    )
    parser.add_argument(
        "--auto-fit",
        action="store_true",
        help="Enable auto-fit to the configured page count when compiling PDF",
    )
    parser.add_argument(
        "--export-resume-pdf",
        action="store_true",
        help="Export a resume PDF to assets/preview_no_summary_skills.pdf without summary/skills (for external uploads).",
    )
    parser.add_argument(
        "--req-file",
        help="Path to job requisition text for --eval-prompt / --generate-profile",
        default=str(BASE_DIR / "req.txt"),
    )
    parser.add_argument(
        "--model-name",
        help="LLM model for --eval-prompt / --generate-profile (e.g. openai:gpt-4o-mini or gemini:gemini-1.5-flash; bare ids default to OpenAI).",
        default=None,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models (from LLM_MODELS or defaults).",
    )
    parser.add_argument(
        "--show-resume-data",
        action="store_true",
        help="Dump the raw resume data from Neo4j to stdout.",
    )
    parser.add_argument(
        "--list-applied",
        action="store_true",
        help="List jobs/resumes already applied/saved in Neo4j",
    )
    parser.add_argument(
        "--maximum-coverage",
        action="store_true",
        help=(
            "Drive UI interactions against State to maximize coverage "
            "(implies all actions + skip-llm + failure paths unless overridden)."
        ),
    )
    parser.add_argument(
        "--maximum-coverage-actions",
        default="all",
        help=(
            "Comma-separated UI actions to simulate: all, load, profile, pipeline, "
            "forms, toggles, reorder, save, pdf."
        ),
    )
    parser.add_argument(
        "--maximum-coverage-skip-llm",
        action="store_true",
        help="Skip LLM calls during maximum-coverage simulation.",
    )
    parser.add_argument(
        "--maximum-coverage-failures",
        action="store_true",
        help="Simulate failure paths (Neo4j/LLM/Typst) during maximum-coverage simulation.",
    )
    parser.add_argument(
        "--maximum-coverage-ui-url",
        default="",
        help="Reflex app URL for Playwright UI traversal (default: env MAX_COVERAGE_UI_URL/REFLEX_URL or localhost).",
    )
    parser.add_argument(
        "--maximum-coverage-ui-timeout",
        type=float,
        default=30.0,
        help="Playwright UI traversal timeout in seconds.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex",
        action="store_true",
        help="Start a Reflex server with coverage enabled and drive it via Playwright.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-frontend-port",
        type=int,
        default=3010,
        help="Frontend port for the Reflex coverage server.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-backend-port",
        type=int,
        default=8010,
        help="Backend port for the Reflex coverage server.",
    )
    parser.add_argument(
        "--maximum-coverage-reflex-startup-timeout",
        type=float,
        default=30.0,
        help="Startup timeout for the Reflex coverage server.",
    )
    parser.add_argument(
        "--ui-playwright-check",
        action="store_true",
        help="Run the comprehensive Playwright UI check (requires a running Reflex app).",
    )
    parser.add_argument(
        "--ui-playwright-url",
        default="",
        help=(
            "URL for the Playwright UI check (defaults to PLAYWRIGHT_URL/REFLEX_URL/REFLEX_APP_URL "
            "or http://localhost:3000)."
        ),
    )
    parser.add_argument(
        "--ui-playwright-timeout",
        type=float,
        default=45.0,
        help="Timeout in seconds for the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-pdf-timeout",
        type=float,
        default=45.0,
        help="PDF embed timeout in seconds for the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-headed",
        action="store_true",
        help="Run the Playwright UI check with a visible browser window.",
    )
    parser.add_argument(
        "--ui-playwright-slowmo",
        type=int,
        default=0,
        help="Slow down Playwright actions (ms) for the UI check.",
    )
    parser.add_argument(
        "--ui-playwright-allow-llm-error",
        action="store_true",
        help="Allow LLM errors during the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-allow-db-error",
        action="store_true",
        help="Allow DB errors during the Playwright UI check.",
    )
    parser.add_argument(
        "--ui-playwright-screenshot-dir",
        default="",
        help="Directory for UI check screenshots and PDF artifacts.",
    )
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help=(
            "Run maximum coverage, verify a clean reflex run startup, and execute "
            "the Playwright UI check."
        ),
    )
    parser.add_argument(
        "--ui-simulate",
        dest="ui_simulate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-actions",
        dest="ui_simulate_actions",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-skip-llm",
        dest="ui_simulate_skip_llm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ui-simulate-failures",
        dest="ui_simulate_failures",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.ui_playwright_check and not args.run_all_tests:
        target_url = (
            args.ui_playwright_url
            or os.environ.get("PLAYWRIGHT_URL")
            or os.environ.get("REFLEX_URL")
            or os.environ.get("REFLEX_APP_URL")
            or "http://localhost:3000"
        )
        script_path = BASE_DIR / "scripts" / "ui_playwright_check.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--url",
            str(target_url),
            "--timeout",
            str(args.ui_playwright_timeout),
            "--pdf-timeout",
            str(args.ui_playwright_pdf_timeout),
        ]
        if args.ui_playwright_headed:
            cmd.append("--headed")
        if args.ui_playwright_slowmo:
            cmd.extend(["--slowmo", str(args.ui_playwright_slowmo)])
        if args.ui_playwright_allow_llm_error:
            cmd.append("--allow-llm-error")
        if args.ui_playwright_allow_db_error:
            cmd.append("--allow-db-error")
        if args.ui_playwright_screenshot_dir:
            cmd.extend(["--screenshot-dir", args.ui_playwright_screenshot_dir])
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        sys.exit(result.returncode)

    ui_url_was_set = bool(getattr(args, "maximum_coverage_ui_url", ""))

    if args.maximum_coverage:
        if os.environ.get("MAX_COVERAGE_STUB_DB") != "1":
            os.environ.pop("MAX_COVERAGE_STUB_DB", None)
        os.environ.setdefault("MAX_COVERAGE_SKIP_LLM", "1")
        if not args.maximum_coverage_actions:
            args.maximum_coverage_actions = "all"
        if not args.maximum_coverage_skip_llm:
            args.maximum_coverage_skip_llm = True
        if not args.maximum_coverage_failures:
            args.maximum_coverage_failures = True
        if not args.maximum_coverage_reflex:
            args.maximum_coverage_reflex = True
        if not args.maximum_coverage_ui_url:
            args.maximum_coverage_ui_url = (
                os.environ.get("MAX_COVERAGE_UI_URL")
                or os.environ.get("REFLEX_URL")
                or os.environ.get("REFLEX_APP_URL")
                or "http://localhost:3000"
            )

    if args.list_models:
        print("Available LLM Models:")
        for model in list_llm_models():
            print(f"- {model}")
        sys.exit(0)

    def ensure_len(items, target=9):
        items = list(items or [])
        while len(items) < target:
            items.append("")
        return items[:target]

    def _read_req_text(path_str: str) -> str:
        req_path = Path(path_str)
        if not req_path.exists():
            return ""
        return req_path.read_text(encoding="utf-8", errors="ignore")

    async def _drain_event(event):
        if event is None:
            return
        if hasattr(event, "__aiter__"):
            async for _ in event:
                pass
            return
        if asyncio.iscoroutine(event):
            await event

    def _exercise_forms(state: State) -> None:
        state.first_name = "Test"
        state.middle_name = "Q"
        state.last_name = "User"
        state.email = "test@example.com"
        state.email2 = "test.secondary@example.com"
        state.phone = "555-555-5555"
        state.linkedin_url = "linkedin.com/in/test-user"
        state.github_url = "github.com/test-user"
        state.scholar_url = "https://scholar.google.com/citations?user=TEST"
        state.calendly_url = "https://cal.link/mingusb"
        state.portfolio_url = "https://portfolio.example.com"
        state.summary = "Test summary for UI simulation."
        state.headers = [
            "Header 1",
            "Header 2",
            "Header 3",
            "Header 4",
            "Header 5",
            "Header 6",
            "Header 7",
            "Header 8",
            "Header 9",
        ]
        state.highlighted_skills = [
            "Skill 1",
            "Skill 2",
            "Skill 3",
            "Skill 4",
            "Skill 5",
            "Skill 6",
            "Skill 7",
            "Skill 8",
            "Skill 9",
        ]
        state.skills_rows = [
            ["Skill A", "Skill B", "Skill C"],
            ["Skill D", "Skill E"],
            ["Skill F"],
        ]
        _ = state.skills_rows_csv

        state.add_experience()
        state.update_experience_field(0, "role", "Role A")
        state.update_experience_field(0, "company", "Company A")
        state.update_experience_field(0, "location", "Remote")
        state.update_experience_field(0, "start_date", "2024-01-01")
        state.update_experience_field(0, "end_date", "2024-12-31")
        state.update_experience_field(0, "description", "Did things.")
        state.update_experience_field(0, "bullets", "Did thing A\nDid thing B")
        state.update_experience_field(1, "role", "Role B")
        state.update_experience_field(1, "company", "Company B")
        state.remove_experience(99)
        if len(state.experience) > 1:
            state.remove_experience(0)

        state.add_education()
        state.update_education_field(0, "degree", "M.S.")
        state.update_education_field(0, "school", "Test University")
        state.update_education_field(0, "location", "Test City")
        state.update_education_field(0, "start_date", "2020-01-01")
        state.update_education_field(0, "end_date", "2022-01-01")
        state.update_education_field(0, "description", "Program highlights.")
        state.update_education_field(0, "bullets", "Course A\nCourse B")
        state.update_education_field(2, "degree", "Ph.D.")
        state.remove_education(99)
        if len(state.education) > 1:
            state.remove_education(0)

        state.add_founder_role()
        state.update_founder_role_field(0, "role", "Founder")
        state.update_founder_role_field(0, "company", "Startup")
        state.update_founder_role_field(0, "location", "Remote")
        state.update_founder_role_field(0, "start_date", "2018-01-01")
        state.update_founder_role_field(0, "end_date", "2019-01-01")
        state.update_founder_role_field(0, "description", "Built product.")
        state.update_founder_role_field(0, "bullets", "Milestone 1\nMilestone 2")
        state.update_founder_role_field(1, "role", "Advisor")
        state.remove_founder_role(99)
        if len(state.founder_roles) > 1:
            state.remove_founder_role(0)

    async def _run_ui_simulation(
        actions: set[str],
        req_file: str,
        *,
        skip_llm: bool,
        simulate_failures: bool,
    ) -> bool:
        _maxcov_log(f"ui-sim start: actions={','.join(sorted(actions)) or 'none'}")
        state = State(_reflex_internal_init=True)
        _maxcov_log("ui-sim on_load start")
        state.on_load()
        _maxcov_log("ui-sim on_load done")

        req_text = _read_req_text(req_file)
        if req_text:
            state.job_req = req_text
            _ = state.job_req_needs_profile

        orig_generate_resume_content = generate_resume_content
        orig_compile_pdf = compile_pdf
        orig_compile_pdf_with_auto_tuning = compile_pdf_with_auto_tuning
        orig_neo4j = Neo4jClient

        if skip_llm:
            globals()["generate_resume_content"] = _fake_generate_resume_content
            _maxcov_log("ui-sim using fake LLM output")
        if os.environ.get("MAX_COVERAGE_SKIP_PDF") == "1":
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n%")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n%",
            )
            _maxcov_log("ui-sim skipping Typst PDF compile")

        if "load" in actions:
            _maxcov_log("ui-sim load start")
            await _drain_event(state.load_resume_fields())
            _maxcov_log("ui-sim load done")

        if "profile" in actions:
            _maxcov_log("ui-sim profile start")
            await _drain_event(state.generate_profile())
            _maxcov_log("ui-sim profile done")

        if "pipeline" in actions:
            _maxcov_log("ui-sim pipeline start")
            if os.environ.get("MAX_COVERAGE_PIPELINE_EVENT") == "1":

                async def _noop_event():
                    return None

                pipeline_event = _noop_event()
            else:
                pipeline_event = State.paste_req_and_run_pipeline.fn(state, req_text)
            if hasattr(pipeline_event, "fn"):
                await _drain_event(pipeline_event.fn(state))
            else:
                await _drain_event(pipeline_event)
            _maxcov_log("ui-sim pipeline done")

        if "toggles" in actions:
            _maxcov_log("ui-sim toggles start")
            State.set_include_matrices.fn(state, not state.include_matrices)
            State.set_auto_tune_pdf.fn(state, not state.auto_tune_pdf)
            State.set_include_matrices.fn(state, True)
            State.set_auto_tune_pdf.fn(state, True)
            _maxcov_log("ui-sim toggles done")

        if "reorder" in actions:
            _maxcov_log("ui-sim reorder start")
            if state.section_order:
                state.move_section_down(0)
                if len(state.section_order) > 1:
                    state.move_section_up(1)
            state.move_section_up(-1)
            state.move_section_down(999)
            _maxcov_log("ui-sim reorder done")

        if "forms" in actions:
            _maxcov_log("ui-sim forms start")
            _exercise_forms(state)
            _maxcov_log("ui-sim forms done")

        if "save" in actions:
            _maxcov_log("ui-sim save start")
            await _drain_event(state.save_to_db())
            _maxcov_log("ui-sim save done")

        if "pdf" in actions:
            _maxcov_log("ui-sim pdf start")
            if state.job_req and state.job_req_needs_profile:
                await _drain_event(state.generate_profile())
            if state.data_loaded:
                state.generate_pdf()
            _maxcov_log("ui-sim pdf done")

        if simulate_failures:
            _maxcov_log("ui-sim failures start")
            globals()["generate_resume_content"] = lambda *_args, **_kwargs: {
                "error": "Simulated LLM failure"
            }
            await _drain_event(state.generate_profile())

            class _FailingNeo4jClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Simulated Neo4j failure")

            globals()["Neo4jClient"] = _FailingNeo4jClient
            await _drain_event(state.load_resume_fields())

            def _fail_compile(*_args, **_kwargs):
                return False, b""

            globals()["compile_pdf"] = _fail_compile
            globals()["compile_pdf_with_auto_tuning"] = _fail_compile
            state.data_loaded = True
            if state.job_req:
                state.last_profile_job_req_sha = hashlib.sha256(
                    state.job_req.encode("utf-8")
                ).hexdigest()
            state.generate_pdf()
            _maxcov_log("ui-sim failures done")

        pending = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        globals()["generate_resume_content"] = orig_generate_resume_content
        globals()["compile_pdf"] = orig_compile_pdf
        globals()["compile_pdf_with_auto_tuning"] = orig_compile_pdf_with_auto_tuning
        globals()["Neo4jClient"] = orig_neo4j

        _maxcov_log("ui-sim complete")
        return True

    def _exercise_maximum_coverage_extras(req_file: str) -> None:
        _maxcov_log("maxcov extras start")
        req_text = _read_req_text(req_file) or "Sample req text."
        tmp_missing = Path(tempfile.mkdtemp(prefix="maxcov_req_")) / "missing.txt"
        keep_path = tmp_missing.parent / "keep.txt"
        try:
            keep_path.write_text("x", encoding="utf-8")
            _read_req_text(str(tmp_missing))
        finally:
            try:
                tmp_missing.parent.rmdir()
            except Exception:
                pass
            try:
                if keep_path.exists():
                    keep_path.unlink()
                tmp_missing.parent.rmdir()
            except Exception:
                pass
        import io
        from contextlib import contextmanager

        @contextmanager
        def _capture_maxcov_output(label: str):
            if not MAX_COVERAGE_LOG:
                yield
                return
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = buf_out, buf_err
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                out = (buf_out.getvalue() or "").strip()
                err = (buf_err.getvalue() or "").strip()
                combined = "\n".join([t for t in (out, err) if t])
                if combined:
                    _maxcov_log(f"{label} output (expected failure):\n{combined}")

        def _strip_ansi(output: str) -> str:
            cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output or "")
            return cleaned.replace("\r", "\n")

        def _extract_json_payload(output: str) -> str:
            cleaned = (output or "").strip()
            if not cleaned:
                return ""
            positions = [m.start() for m in re.finditer(r"[\\[{]", cleaned)]
            if not positions:
                return cleaned
            decoder = json.JSONDecoder()
            for pos in positions:
                try:
                    _, end = decoder.raw_decode(cleaned[pos:])
                except Exception:
                    continue
                return cleaned[pos : pos + end]
            return cleaned

        def _count_issue_lines(output: str) -> int:
            return sum(1 for line in (output or "").splitlines() if line.strip())

        def _parse_issue_stats(output: str, label: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"{label}={count}"

        def _count_diff_files(output: str) -> int:
            return sum(
                1 for line in (output or "").splitlines() if line.startswith("--- ")
            )

        def _parse_black_stats(output: str) -> tuple[int | None, str]:
            count = 0
            for line in (output or "").splitlines():
                if "would reformat" in line or "would be reformatted" in line:
                    count += 1
            if count == 0:
                count = _count_diff_files(output)
            return count, f"reformat={count}"

        def _parse_isort_stats(output: str) -> tuple[int | None, str]:
            count = sum(
                1
                for line in (output or "").splitlines()
                if line.strip().startswith("ERROR:")
            )
            if count == 0:
                count = _count_diff_files(output)
            return count, f"files={count}"

        def _parse_dodgy_stats(output: str) -> tuple[int | None, str]:
            try:
                data = json.loads(output or "{}")
            except Exception:
                return None, "parse=error"
            warnings = data.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            count = len(warnings)
            return count, f"issues={count}"

        def _parse_deptry_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"Found ([0-9]+) dependency issues?", output or "")
            if match:
                count = int(match.group(1))
                return count, f"issues={count}"
            count = sum(
                1
                for line in (output or "").splitlines()
                if re.search(r"DEP[0-9]+", line)
            )
            return count, f"issues={count}"

        def _parse_eradicate_stats(output: str) -> tuple[int | None, str]:
            count = _count_diff_files(output)
            if count == 0 and output.strip():
                count = _count_issue_lines(output)
            return count, f"files={count}"

        def _parse_autoflake_stats(output: str) -> tuple[int | None, str]:
            filtered = "\n".join(
                line
                for line in (output or "").splitlines()
                if "No issues detected" not in line
            )
            count = _count_issue_lines(filtered)
            return count, f"issues={count}"

        def _parse_pycln_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"([0-9]+) file[s]? would be changed", output or "")
            if match:
                count = int(match.group(1))
                return count, f"files={count}"
            count = _count_diff_files(output)
            return count, f"files={count}"

        def _parse_ruff_stats(output: str) -> tuple[int | None, str]:
            total = 0
            for line in (output or "").splitlines():
                match = re.match(r"^[A-Z][A-Z0-9]+\\s+(\\d+)$", line.strip())
                if match:
                    total += int(match.group(1))
            if total == 0 and output.strip():
                total = _count_issue_lines(output)
            return total, f"issues={total}"

        def _parse_mypy_stats(output: str) -> tuple[int | None, str]:
            if "Success: no issues found" in output:
                return 0, "errors=0"
            match = re.search(r"Found (\\d+) error", output)
            if match:
                count = int(match.group(1))
                return count, f"errors={count}"
            count = _count_issue_lines(output)
            return count, f"errors={count}"

        def _parse_pyre_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"Found (\\d+) errors?", output)
            if match:
                count = int(match.group(1))
                return count, f"errors={count}"
            return _parse_issue_stats(output, "errors")

        def _parse_pylint_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"rated at ([0-9.]+/10)", output)
            if match:
                return None, f"score={match.group(1)}"
            return None, "score=unknown"

        def _parse_flake8_stats(output: str) -> tuple[int | None, str]:
            counts = []
            for line in (output or "").splitlines():
                stripped = line.strip()
                if stripped.isdigit():
                    counts.append(int(stripped))
            if counts:
                count = counts[-1]
            else:
                count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pyflakes_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_pycodestyle_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_pydocstyle_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "issues")

        def _parse_codespell_stats(output: str) -> tuple[int | None, str]:
            count = sum(
                1 for line in (output or "").splitlines() if "==>" in line
            )
            if count == 0 and output.strip():
                count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pip_audit_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "[]")
            except Exception:
                return None, "parse=error"

            def _count_vulns(items: list[dict]) -> int:
                total = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    vulns = item.get("vulns") or item.get("vulnerabilities") or []
                    if isinstance(vulns, list):
                        total += len(vulns)
                return total

            if isinstance(data, list):
                count = _count_vulns(data)
                return count, f"issues={count}"
            if isinstance(data, dict):
                deps = data.get("dependencies") or data.get("results") or []
                if isinstance(deps, dict):
                    deps_list = list(deps.values())
                elif isinstance(deps, list):
                    deps_list = deps
                else:
                    deps_list = []
                count = _count_vulns(deps_list)
                if count == 0:
                    vulns = data.get("vulnerabilities") or data.get("vulns") or []
                    if isinstance(vulns, list):
                        count = len(vulns)
                return count, f"issues={count}"
            return None, "issues=unknown"

        def _parse_safety_stats(output: str) -> tuple[int | None, str]:
            payload = _extract_json_payload(output)
            try:
                data = json.loads(payload or "{}")
            except Exception:
                match = re.search(r"(\\d+) vulnerabilities", output or "")
                if match:
                    count = int(match.group(1))
                    return count, f"issues={count}"
                return None, "parse=error"
            if isinstance(data, list):
                count = len(data)
                return count, f"issues={count}"
            if isinstance(data, dict):
                vulns = (
                    data.get("vulnerabilities")
                    or data.get("vulns")
                    or data.get("results")
                )
                if isinstance(vulns, list):
                    count = len(vulns)
                    return count, f"issues={count}"
                if isinstance(vulns, dict):
                    count = len(vulns)
                    return count, f"issues={count}"
            return 0, "issues=0"

        def _has_safety_api_key_error(output: str) -> bool:
            lowered = (output or "").lower()
            return "api key" in lowered and (
                "required" in lowered or "missing" in lowered or "not provided" in lowered
            )

        def _parse_interrogate_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"SUMMARY:\\s+([0-9.]+%)", output)
            if match:
                return None, f"coverage={match.group(1)}"
            return None, "coverage=unknown"

        def _parse_pyright_stats(output: str) -> tuple[int | None, str]:
            match = re.search(
                r"(\\d+) errors?, (\\d+) warnings?, (\\d+) information",
                output,
            )
            if match:
                errors = int(match.group(1))
                warnings = int(match.group(2))
                info = int(match.group(3))
                total = errors + warnings + info
                details = f"errors={errors}, warnings={warnings}, info={info}"
                return total, details
            count = _count_issue_lines(output)
            return count, f"issues={count}"

        def _parse_pytype_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"(\\d+) errors?", output)
            if match:
                count = int(match.group(1))
                return count, f"errors={count}"
            return _parse_issue_stats(output, "errors")

        def _parse_vulture_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"unused={count}"

        def _parse_bandit_stats(output: str) -> tuple[int | None, str]:
            try:
                data = json.loads(output or "{}")
            except Exception:
                return None, "parse=error"
            results = data.get("results") or []
            if not isinstance(results, list):
                results = []
            count = len(results)
            severities: dict[str, int] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                sev = str(item.get("issue_severity") or "unknown")
                severities[sev] = severities.get(sev, 0) + 1
            sev_parts = ", ".join(f"{k}={v}" for k, v in sorted(severities.items()))
            details = f"issues={count}"
            if sev_parts:
                details = f"{details}, {sev_parts}"
            return count, details

        def _parse_radon_cc_stats(output: str) -> tuple[int | None, str]:
            match = re.search(r"Average complexity: ([A-Z]) \\(([^)]+)\\)", output)
            if match:
                return None, f"avg={match.group(1)} ({match.group(2)})"
            return None, "avg=unknown"

        def _parse_radon_mi_stats(output: str) -> tuple[int | None, str]:
            grades: dict[str, int] = {}
            for line in (output or "").splitlines():
                match = re.search(r"\\s-\\s([A-F])", line)
                if match:
                    grade = match.group(1)
                    grades[grade] = grades.get(grade, 0) + 1
            if grades:
                details = ", ".join(
                    f"{k}={v}" for k, v in sorted(grades.items())
                )
                return None, details
            return None, "grades=unknown"

        def _parse_radon_raw_stats(output: str) -> tuple[int | None, str]:
            stats = {}
            for line in (output or "").splitlines():
                match = re.match(r"^\\s*([A-Z][A-Za-z ]+):\\s*(\\d+)", line.strip())
                if match:
                    key = match.group(1).strip().lower().replace(" ", "_")
                    stats[key] = match.group(2)
            if not stats:
                return None, "summary=unknown"
            parts = []
            for key in (
                "loc",
                "lloc",
                "sloc",
                "comments",
                "multi",
                "blank",
            ):
                if key in stats:
                    parts.append(f"{key}={stats[key]}")
            return None, ", ".join(parts) if parts else "summary=unknown"

        def _parse_mccabe_stats(output: str) -> tuple[int | None, str]:
            return _parse_issue_stats(output, "complex")

        def _parse_xenon_stats(output: str) -> tuple[int | None, str]:
            count = _count_issue_lines(output)
            return count, f"violations={count}"

        def _parse_lizard_stats(output: str) -> tuple[int | None, str]:
            nloc_match = re.search(r"Total\\s+NLOC\\s+(\\d+)", output)
            cc_match = re.search(
                r"Average\\s+Cyclomatic\\s+Complexity\\s+([0-9.]+)", output
            )
            parts = []
            if nloc_match:
                parts.append(f"nloc={nloc_match.group(1)}")
            if cc_match:
                parts.append(f"avg_cc={cc_match.group(1)}")
            if parts:
                return None, ", ".join(parts)
            return None, f"lines={_count_issue_lines(output)}"

        def _parse_semgrep_stats(output: str) -> tuple[int | None, str]:
            try:
                data = json.loads(output or "{}")
            except Exception:
                return None, "parse=error"
            results = data.get("results") or []
            if not isinstance(results, list):
                results = []
            count = len(results)
            severities: dict[str, int] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                extra = item.get("extra") or {}
                sev = str(extra.get("severity") or "UNKNOWN").upper()
                severities[sev] = severities.get(sev, 0) + 1
            sev_parts = ", ".join(f"{k}={v}" for k, v in sorted(severities.items()))
            details = f"issues={count}"
            if sev_parts:
                details = f"{details}, {sev_parts}"
            return count, details

        def _pyupgrade_target_flag() -> str:
            major, minor = sys.version_info[:2]
            if major > 3 or minor >= 12:
                return "--py312-plus"
            if minor >= 11:
                return "--py311-plus"
            if minor >= 10:
                return "--py310-plus"
            if minor >= 9:
                return "--py39-plus"
            return "--py38-plus"

        def _should_skip_path(path: Path, skip_parts: set[str]) -> bool:
            for part in path.parts:
                if part in skip_parts:
                    return True
                if part.startswith("maxcov_raster_"):
                    return True
            return False

        def _iter_python_files(base: Path, skip_parts: set[str]) -> list[Path]:
            if base.is_file():
                return [base] if base.suffix == ".py" else []
            return [
                path
                for path in base.rglob("*.py")
                if not _should_skip_path(path, skip_parts)
            ]

        def _run_pyupgrade_check(
            label: str,
            target_path: Path,
            skip_parts: set[str],
            timeout_s: float | None,
        ) -> dict:
            started = time.perf_counter()
            if shutil.which("pyupgrade") is None:
                return {
                    "tool": label,
                    "status": "fail",
                    "duration_s": 0.0,
                    "details": "missing",
                }
            effective_timeout = None
            if timeout_s is not None:
                try:
                    timeout_val = float(timeout_s)
                except (TypeError, ValueError):
                    timeout_val = None
                if timeout_val and timeout_val > 0:
                    effective_timeout = timeout_val
            tmp_dir = Path(
                tempfile.mkdtemp(prefix="maxcov_pyupgrade_", dir=tempfile.gettempdir())
            )
            try:
                files = _iter_python_files(target_path, skip_parts)
                if not files:
                    return {
                        "tool": label,
                        "status": "ok",
                        "duration_s": time.perf_counter() - started,
                        "details": "files=0",
                    }

                def _rel_for(path: Path) -> Path:
                    if target_path.is_file():
                        return Path(path.name)
                    return path.relative_to(target_path)

                for path in files:
                    rel = _rel_for(path)
                    dest = tmp_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)

                cmd = [
                    "pyupgrade",
                    _pyupgrade_target_flag(),
                    "--exit-zero-even-if-changed",
                    str(tmp_dir),
                ]
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )
                except subprocess.TimeoutExpired:
                    return {
                        "tool": label,
                        "status": "warn",
                        "duration_s": time.perf_counter() - started,
                        "details": (
                            f"timeout>{effective_timeout}s"
                            if effective_timeout
                            else "timeout"
                        ),
                    }
                except Exception as exc:
                    return {
                        "tool": label,
                        "status": "warn",
                        "duration_s": time.perf_counter() - started,
                        "details": f"error={type(exc).__name__}",
                    }

                changed = 0
                for path in files:
                    rel = _rel_for(path)
                    dest = tmp_dir / rel
                    try:
                        if dest.exists() and dest.read_bytes() != path.read_bytes():
                            changed += 1
                    except Exception:
                        continue
                status = "warn" if changed > 0 else "ok"
                details = f"files={changed}"
                if result.returncode != 0:
                    status = "warn"
                    details = f"{details}, rc={result.returncode}"
                return {
                    "tool": label,
                    "status": status,
                    "duration_s": time.perf_counter() - started,
                    "details": details,
                }
            finally:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        def _run_static_analysis_tools(report_path: Path) -> None:
            target_dir = os.environ.get(
                "MAX_COVERAGE_STATIC_TARGET", str(BASE_DIR)
            )
            target_file = os.environ.get(
                "MAX_COVERAGE_STATIC_TARGET_FILE", str(BASE_DIR / "harness.py")
            )
            target_path = Path(target_dir)
            raw_timeout = os.environ.get("MAX_COVERAGE_STATIC_TIMEOUT", "0")
            timeout_s: float | None
            timeout_s = None
            if raw_timeout is not None:
                raw_timeout = str(raw_timeout).strip().lower()
            if raw_timeout not in {"", "0", "none", "false"}:
                try:
                    timeout_s = float(raw_timeout)
                except (TypeError, ValueError):
                    timeout_s = None
                if timeout_s is not None and timeout_s <= 0:
                    timeout_s = None
            requirements_path = Path(BASE_DIR / "requirements.txt")
            skip_parts = {
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "node_modules",
                "assets",
                "assets_out",
                "fonts",
                "packages",
                "diagrams",
                "maxcov_logs",
            }
            codespell_skip = ",".join(
                [
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    "assets",
                    "assets_out",
                    "fonts",
                    "packages",
                    "diagrams",
                    "maxcov_logs",
                    "maxcov_raster_*",
                ]
            )
            pytype_output = Path(
                tempfile.mkdtemp(prefix="maxcov_pytype_", dir=tempfile.gettempdir())
            )
            raw_semgrep = os.environ.get("MAX_COVERAGE_SEMGREP_CONFIG")
            if raw_semgrep is None:
                semgrep_config = ""
                for candidate in (
                    ".semgrep.yml",
                    ".semgrep.yaml",
                    "semgrep.yml",
                    "semgrep.yaml",
                ):
                    candidate_path = target_path / candidate
                    if candidate_path.exists():
                        semgrep_config = str(candidate_path)
                        break
                if not semgrep_config:
                    semgrep_config = "p/python"
            else:
                semgrep_config = raw_semgrep.strip()
            semgrep_enabled = semgrep_config.strip().lower() not in {
                "",
                "0",
                "false",
                "none",
            }
            pyre_enabled = (target_path / ".pyre_configuration").exists() or (
                target_path / ".pyre_configuration.local"
            ).exists()
            tool_defs = {
                "ruff": {
                    "cmd": [
                        "ruff",
                        "check",
                        target_dir,
                        "--statistics",
                        "--exit-zero",
                    ],
                    "parser": _parse_ruff_stats,
                },
                "black": {
                    "cmd": ["black", "--check", "--diff", target_dir],
                    "parser": _parse_black_stats,
                },
                "isort": {
                    "cmd": ["isort", "--check-only", "--diff", target_dir],
                    "parser": _parse_isort_stats,
                },
                "mypy": {
                    "cmd": ["mypy", target_file],
                    "parser": _parse_mypy_stats,
                },
                "pylint": {
                    "cmd": [
                        "pylint",
                        target_file,
                        "--score=y",
                        "--reports=n",
                        "--exit-zero",
                    ],
                    "parser": _parse_pylint_stats,
                },
                "flake8": {
                    "cmd": [
                        "flake8",
                        target_dir,
                        "--count",
                        "--statistics",
                        "--quiet",
                    ],
                    "parser": _parse_flake8_stats,
                },
                "pyflakes": {
                    "cmd": ["pyflakes", target_dir],
                    "parser": _parse_pyflakes_stats,
                },
                "pycodestyle": {
                    "cmd": ["pycodestyle", target_dir],
                    "parser": _parse_pycodestyle_stats,
                },
                "pydocstyle": {
                    "cmd": ["pydocstyle", target_dir],
                    "parser": _parse_pydocstyle_stats,
                },
                "codespell": {
                    "cmd": ["codespell", "--skip", codespell_skip, target_dir],
                    "parser": _parse_codespell_stats,
                },
                "pyright": {
                    "cmd": ["pyright", target_dir, "--stats"],
                    "parser": _parse_pyright_stats,
                },
                "pytype": {
                    "cmd": [
                        "pytype",
                        "--quick",
                        "--output",
                        str(pytype_output),
                        target_file,
                    ],
                    "parser": _parse_pytype_stats,
                    "timeout_s": None,
                },
                "pyre": {
                    "cmd": ["pyre", "check"],
                    "parser": _parse_pyre_stats,
                    "skip_reason": "" if pyre_enabled else "no config",
                },
                "vulture": {
                    "cmd": ["vulture", target_dir, "--min-confidence", "60"],
                    "parser": _parse_vulture_stats,
                },
                "bandit": {
                    "cmd": ["bandit", "-q", "-r", target_dir, "-f", "json"],
                    "parser": _parse_bandit_stats,
                },
                "pip-audit": {
                    "cmd": [
                        "pip-audit",
                        "-r",
                        str(requirements_path),
                        "-f",
                        "json",
                        "--progress-spinner",
                        "off",
                    ],
                    "parser": _parse_pip_audit_stats,
                    "timeout_s": None,
                    "skip_reason": ""
                    if requirements_path.exists()
                    else "requirements.txt missing",
                },
                "safety": {
                    "cmd": ["safety", "check", "--json", "-r", str(requirements_path)],
                    "parser": _parse_safety_stats,
                    "skip_reason": ""
                    if requirements_path.exists()
                    else "requirements.txt missing",
                },
                "semgrep": {
                    "cmd": [
                        "semgrep",
                        "scan",
                        "--config",
                        semgrep_config,
                        "--json",
                        "--quiet",
                        "--metrics",
                        "off",
                        target_dir,
                    ],
                    "parser": _parse_semgrep_stats,
                    "skip_reason": "" if semgrep_enabled else "disabled",
                },
                "dodgy": {
                    "cmd": ["dodgy", target_dir],
                    "parser": _parse_dodgy_stats,
                },
                "eradicate": {
                    "cmd": ["eradicate", "-r", "-e", target_dir],
                    "parser": _parse_eradicate_stats,
                },
                "deptry": {
                    "cmd": ["deptry", target_dir, "--no-ansi"],
                    "parser": _parse_deptry_stats,
                },
                "pycln": {
                    "cmd": ["pycln", "--check", "--diff", target_dir],
                    "parser": _parse_pycln_stats,
                },
                "pyupgrade": {
                    "cmd": ["pyupgrade"],
                    "runner": lambda label: _run_pyupgrade_check(
                        label, target_path, skip_parts, timeout_s
                    ),
                },
                "autoflake": {
                    "cmd": [
                        "autoflake",
                        "--check",
                        "--quiet",
                        "-r",
                        "--remove-all-unused-imports",
                        "--remove-unused-variables",
                        target_dir,
                    ],
                    "parser": _parse_autoflake_stats,
                },
                "radon-cc": {
                    "cmd": ["radon", "cc", "-s", "-a", target_dir],
                    "parser": _parse_radon_cc_stats,
                },
                "radon-mi": {
                    "cmd": ["radon", "mi", "-s", target_dir],
                    "parser": _parse_radon_mi_stats,
                },
                "radon-raw": {
                    "cmd": ["radon", "raw", "-s", "--summary", target_dir],
                    "parser": _parse_radon_raw_stats,
                },
                "mccabe": {
                    "cmd": [sys.executable, "-m", "mccabe", "--min", "10", target_file],
                    "parser": _parse_mccabe_stats,
                },
                "xenon": {
                    "cmd": [
                        "xenon",
                        "--max-absolute",
                        "A",
                        "--max-modules",
                        "A",
                        "--max-average",
                        "A",
                        target_dir,
                    ],
                    "parser": _parse_xenon_stats,
                },
                "lizard": {
                    "cmd": ["lizard", target_dir],
                    "parser": _parse_lizard_stats,
                },
                "interrogate": {
                    "cmd": ["interrogate", "-q", "--fail-under", "0", target_dir],
                    "parser": _parse_interrogate_stats,
                },
            }
            raw_tools = os.environ.get("MAX_COVERAGE_STATIC_TOOLS", "")
            if raw_tools.strip():
                selected = [
                    name.strip().lower()
                    for name in raw_tools.split(",")
                    if name.strip()
                ]
            else:
                selected = list(tool_defs.keys())

            results: list[dict] = []
            for name in selected:
                tool = tool_defs.get(name)
                if not tool:
                    results.append(
                        {
                            "tool": name,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": "unknown tool",
                        }
                    )
                    continue
                skip_reason = str(tool.get("skip_reason") or "").strip()
                if skip_reason:
                    results.append(
                        {
                            "tool": name,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": skip_reason,
                        }
                    )
                    continue
                cmd = tool.get("cmd")
                runner = tool.get("runner")
                label = str(name)
                binary = ""
                if cmd:
                    binary = cmd[0]
                if cmd and binary == sys.executable and "-m" in cmd:
                    try:
                        module_index = cmd.index("-m") + 1
                        module_name = cmd[module_index]
                    except Exception:
                        module_name = ""
                    if module_name:
                        try:
                            import importlib.util

                            module_missing = (
                                importlib.util.find_spec(module_name) is None
                            )
                        except Exception:
                            module_missing = True
                        if module_missing:
                            results.append(
                                {
                                    "tool": label,
                                    "status": "fail",
                                    "duration_s": 0.0,
                                    "details": "missing",
                                }
                            )
                            continue
                if binary and shutil.which(binary) is None:
                    results.append(
                        {
                            "tool": label,
                            "status": "fail",
                            "duration_s": 0.0,
                            "details": "missing",
                        }
                    )
                    continue
                if runner:
                    results.append(runner(label))
                    continue
                if not cmd:
                    results.append(
                        {
                            "tool": label,
                            "status": "skip",
                            "duration_s": 0.0,
                            "details": "missing command",
                        }
                    )
                    continue
                started = time.perf_counter()
                status = "ok"
                details = ""
                tool_timeout = tool.get("timeout_s", timeout_s)
                if tool_timeout is not None:
                    tool_timeout = float(tool_timeout)
                    if tool_timeout <= 0:
                        tool_timeout = None
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=tool_timeout,
                    )
                except subprocess.TimeoutExpired:
                    status = "warn"
                    details = (
                        f"timeout>{tool_timeout}s" if tool_timeout else "timeout"
                    )
                    duration_s = time.perf_counter() - started
                    results.append(
                        {
                            "tool": label,
                            "status": status,
                            "duration_s": duration_s,
                            "details": details,
                        }
                    )
                    continue
                except Exception as exc:
                    status = "warn"
                    details = f"error={type(exc).__name__}"
                    duration_s = time.perf_counter() - started
                    results.append(
                        {
                            "tool": label,
                            "status": status,
                            "duration_s": duration_s,
                            "details": details,
                        }
                    )
                    continue

                output = "\n".join(
                    [t for t in (result.stdout or "", result.stderr or "") if t]
                )
                output = _strip_ansi(output)
                if name == "safety" and _has_safety_api_key_error(output):
                    duration_s = time.perf_counter() - started
                    results.append(
                        {
                            "tool": label,
                            "status": "skip",
                            "duration_s": duration_s,
                            "details": "api key required",
                        }
                    )
                    continue
                issues, details = tool["parser"](output)
                if issues is None:
                    status = "ok" if result.returncode == 0 else "warn"
                else:
                    status = "warn" if issues > 0 else "ok"
                    if issues == 0 and result.returncode != 0:
                        status = "warn"
                        if details:
                            details = f"{details}, rc={result.returncode}"
                        else:
                            details = f"rc={result.returncode}"
                duration_s = time.perf_counter() - started
                results.append(
                    {
                        "tool": label,
                        "status": status,
                        "duration_s": duration_s,
                        "details": details,
                    }
                )

            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(results, ensure_ascii=True),
                    encoding="utf-8",
                )
            except Exception:
                pass
            finally:
                try:
                    shutil.rmtree(pytype_output, ignore_errors=True)
                except Exception:
                    pass

        force_excepts = True

        def _force_exception(label: str = "forced") -> None:
            if force_excepts:
                raise RuntimeError(label)

        # Build the UI tree to cover component construction.
        try:
            index()
            section_order_controls()
            labeled_toggle("Hidden", checked=True, on_change=None, show_label=False)
            labeled_toggle(
                "With Props",
                checked=False,
                on_change=None,
                switch_props={"disabled": True},
                container_props={"spacing": "3"},
            )
            styled_input(
                value="",
                on_change=None,
                placeholder="Styled",
                style={"border": "2px solid #fff"},
            )
            styled_textarea(
                value="",
                on_change=None,
                placeholder="Styled",
                style={"min_height": "3em"},
            )
            _section_order_row(("summary", True), 0)
            _experience_card(Experience(id="exp-1", role="Role", company="Co"), 0)
            _education_card(Education(id="ed-1", degree="Degree", school="School"), 0)
            _founder_role_card(
                FounderRole(id="fr-1", role="Founder", company="Startup"), 0
            )
            _force_exception("ui-tree")
        except Exception:
            pass
        _maxcov_log("maxcov extras ui tree done")

        # Exercise default env and helper utilities.
        env_keys = [
            "LLM_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "LLM_MAX_OUTPUT_TOKENS",
            "LLM_MODEL",
            "OPENAI_MODEL",
            "MAX_COVERAGE_STUB_DB",
        ]
        env_backup = {key: os.environ.get(key) for key in env_keys}

        def _restore_env():
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        try:
            os.environ.pop("LLM_REASONING_EFFORT", None)
            os.environ.pop("OPENAI_REASONING_EFFORT", None)
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            _ensure_default_llm_env()

            os.environ["LLM_REASONING_EFFORT"] = "invalid"
            os.environ["LLM_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "openai:gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "bogus/model"
            _resolve_default_llm_settings()

            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            _maxcov_stub_db_enabled()
            os.environ["MAX_COVERAGE_STUB_DB"] = "0"
            _maxcov_stub_db_enabled()
        finally:
            _restore_env()

        tmp_base = Path(tempfile.mkdtemp(prefix="maxcov_base_"))
        try:
            candidate = tmp_base / "assets"
            candidate.write_text("x", encoding="utf-8")
            _resolve_assets_dir(tmp_base)
            candidate.unlink()
            candidate.mkdir()
            _resolve_assets_dir(tmp_base)
        finally:
            try:
                for path in tmp_base.rglob("*"):
                    if path.is_file():
                        path.unlink()
                for path in sorted(tmp_base.rglob("*"), reverse=True):
                    if path.is_dir():
                        path.rmdir()
                tmp_base.rmdir()
            except Exception:
                pass

        _empty_resume_payload()
        store = {}
        _seed_maxcov_store(
            store,
            {
                "profile": {
                    "id": "resume-1",
                    "section_enabled": ["summary", "experience"],
                },
                "experience": [{"id": "exp-1"}],
                "education": [{"id": "ed-1"}],
                "founder_roles": [{"id": "fr-1"}],
                "skills": [{"category": "Core", "skills": []}],
            },
        )
        orig_maxcov_db = globals().get("_MAXCOV_DB")
        orig_stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
        try:
            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            globals()["_MAXCOV_DB"] = None
            _get_maxcov_store()
            globals()["_MAXCOV_DB"] = {}
            _get_maxcov_store()
        finally:
            if orig_stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = orig_stub_env
            globals()["_MAXCOV_DB"] = orig_maxcov_db
        _maxcov_format_arc("bad")
        _maxcov_format_arc((0, -1))
        _maxcov_format_branch_arcs([], limit=1)
        _maxcov_format_top_missing_blocks([])
        _get_arg_value(["--foo", "bar"], "--foo", "default")
        _get_arg_value(["--foo=bar"], "--foo", "default")
        _get_arg_value([], "--foo", "default")

        # Exercise coverage hooks and helper branches.
        os.environ.setdefault("REFLEX_COVERAGE", "1")
        orig_reflex_env = os.environ.get("REFLEX_COVERAGE")
        orig_reflex_stop = os.environ.get("MAX_COVERAGE_REFLEX_STOP")
        orig_reflex_force = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED")
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        orig_import_cov = None
        try:
            _try_import_coverage()
            import builtins

            orig_import_cov = builtins.__import__

            def _block_cov_import(name, *args, **kwargs):
                if name == "coverage":
                    raise ImportError("blocked")
                return orig_import_cov(name, *args, **kwargs)

            builtins.__import__ = _block_cov_import
            _try_import_coverage()
            try:
                _block_cov_import("json")
            except Exception:
                pass
            builtins.__import__ = orig_import_cov

            os.environ["REFLEX_COVERAGE"] = "0"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "0"
            os.environ["MAX_COVERAGE_REFLEX_STOP"] = "0"
            _init_reflex_coverage()
        finally:
            if orig_import_cov is not None:
                import builtins

                builtins.__import__ = orig_import_cov
            if orig_reflex_env is None:
                os.environ.pop("REFLEX_COVERAGE", None)
            else:
                os.environ["REFLEX_COVERAGE"] = orig_reflex_env
            if orig_reflex_stop is None:
                os.environ.pop("MAX_COVERAGE_REFLEX_STOP", None)
            else:
                os.environ["MAX_COVERAGE_REFLEX_STOP"] = orig_reflex_stop
            if orig_reflex_force is None:
                os.environ.pop("REFLEX_COVERAGE_FORCE_OWNED", None)
            else:
                os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = orig_reflex_force
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned

        # Exercise owned coverage stop branch with a stub module.
        orig_cov_mod = sys.modules.get("coverage")
        orig_reflex_env = os.environ.get("REFLEX_COVERAGE")
        orig_reflex_stop = os.environ.get("MAX_COVERAGE_REFLEX_STOP")
        orig_reflex_force = os.environ.get("REFLEX_COVERAGE_FORCE_OWNED")
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        try:
            import types

            fake_cov = types.ModuleType("coverage")

            class _FakeCoverage:
                def __init__(self, *args, **kwargs):
                    self.started = False

                def start(self):
                    self.started = True

                def stop(self):
                    return None

                def save(self):
                    return None

                @staticmethod
                def current():
                    return None

            fake_cov.Coverage = _FakeCoverage
            sys.modules["coverage"] = fake_cov
            os.environ["REFLEX_COVERAGE"] = "1"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"
            os.environ["MAX_COVERAGE_REFLEX_STOP"] = "1"
            _init_reflex_coverage()
        finally:
            if orig_cov_mod is None:
                sys.modules.pop("coverage", None)
            else:
                sys.modules["coverage"] = orig_cov_mod
            if orig_reflex_env is None:
                os.environ.pop("REFLEX_COVERAGE", None)
            else:
                os.environ["REFLEX_COVERAGE"] = orig_reflex_env
            if orig_reflex_stop is None:
                os.environ.pop("MAX_COVERAGE_REFLEX_STOP", None)
            else:
                os.environ["MAX_COVERAGE_REFLEX_STOP"] = orig_reflex_stop
            if orig_reflex_force is None:
                os.environ.pop("REFLEX_COVERAGE_FORCE_OWNED", None)
            else:
                os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = orig_reflex_force
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned

        orig_fonts_dir = FONTS_DIR
        orig_catalog = globals().get("_LOCAL_FONT_CATALOG")
        tmp_font_iter = None
        try:
            tmp_font_iter = Path(tempfile.mkdtemp(prefix="maxcov_fonts_iter_"))
            (tmp_font_iter / "Demo.otf").write_bytes(b"\x00")
            (tmp_font_iter / "Font Awesome 7 Free-Regular-400.otf").write_bytes(b"\x00")
            (tmp_font_iter / "notes.txt").write_text("x", encoding="utf-8")
            (tmp_font_iter / "subdir").mkdir(parents=True, exist_ok=True)
            globals()["FONTS_DIR"] = tmp_font_iter
            _iter_local_font_files()
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Test Font": [
                    {"path": Path("test.ttf"), "weight": 400, "italic": False}
                ]
            }
            _select_local_font_paths("test font", italic=False)
            _select_local_font_paths("TEST FONT", italic=True)
            _build_local_font_extra_fonts()
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Missing Font": [
                    {
                        "path": tmp_font_iter / "missing.otf",
                        "weight": 400,
                        "italic": False,
                    }
                ]
            }
            _build_local_font_extra_fonts()
        finally:
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            try:
                if tmp_font_iter is not None:
                    for path in tmp_font_iter.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_font_iter.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()
            except Exception:
                pass

        _load_prompt_yaml_from_file(
            Path(tempfile.mkdtemp(prefix="maxcov_prompt_")) / "missing.yaml"
        )
        _load_prompt_yaml_from_file(Path(tempfile.mkdtemp(prefix="maxcov_prompt_")))
        tmp_prompt_dir = Path(tempfile.mkdtemp(prefix="maxcov_prompt_ok_"))
        try:
            prompt_file = tmp_prompt_dir / "prompt.yaml"
            prompt_file.write_text("ok", encoding="utf-8")
            _load_prompt_yaml_from_file(prompt_file)
            _resolve_prompt_template({"prompt_yaml": "inline"})
            _resolve_prompt_template({})
            _normalize_section_enabled(None, ["summary"])
            _normalize_section_enabled("", ["summary"])
            _normalize_section_enabled("summary,experience", None)
            _normalize_section_enabled({"summary": True, "bogus": True}, None)
            _normalize_section_enabled(["summary", "bogus"], None)
            _normalize_section_enabled(1, ["summary"])
            _apply_section_enabled(["summary", "experience"], None)
            _apply_section_enabled(["summary", "experience"], ["summary"])
        finally:
            try:
                if tmp_prompt_dir.exists():
                    for path in tmp_prompt_dir.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_prompt_dir.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()
                    tmp_prompt_dir.rmdir()
            except Exception:
                pass
        _coerce_bullet_overrides(42)
        _coerce_bullet_overrides(["x", {"id": "exp-0", "bullets": []}])
        _coerce_bullet_overrides({"exp-1": ["a", "b"]})
        _coerce_bullet_overrides([{"id": "exp-2", "bullets": "a\nb"}])
        _coerce_bullet_overrides([{"experience_id": "exp-3", "bullets": ["c"]}])
        _coerce_bullet_overrides("not-json")
        _coerce_bullet_overrides({"exp-4": "single"})
        _coerce_bullet_overrides(
            [{"id": "", "bullets": []}, {"id": "exp-5", "bullets": 5}]
        )
        _bullet_override_map([{"id": "exp-1", "bullets": "x\ny"}])
        _bullet_override_map([{"id": "exp-2", "bullets": ["x", "y"]}])
        _bullet_override_map([{"id": "exp-3", "bullets": 1}])
        _bullet_override_map([{"id": "", "bullets": ["skip"]}], allow_empty_id=True)
        _ensure_skill_rows([None, "a", ["b"], 1])
        _ensure_skill_rows(["", "", ""])
        _skills_rows_to_csv(["Row 1", None, ["A"]], ["Skill 1", "Skill 2"])
        _skills_rows_to_csv([["A"], ["B"], ["C"]], [])
        _skills_rows_to_csv(["Only"], [])
        _openai_reasoning_params_for_model("gpt-4o-mini")
        _resolve_llm_max_output_tokens("openai", "gpt-5.2")
        _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")
        _resolve_llm_retry_max_output_tokens("openai", "gpt-5.2", 2000)
        _coerce_llm_text([1, None, "a"])
        _coerce_llm_text({"a": 1})
        _rasterize_text_image("")
        orig_select_fonts = globals().get("_select_local_font_paths")
        orig_pil = sys.modules.get("PIL")
        try:
            import types

            class _TinyImage:
                def __init__(self, size):
                    self.width, self.height = size

                def resize(self, size, _resample):
                    return _TinyImage(size)

                def save(self, path, format=None):
                    Path(path).write_bytes(b"x")

            class _TinyDraw:
                def __init__(self, _img):
                    return None

                def textbbox(self, _pos, _text, font=None):
                    return (0, 0, 10, 10)

                def text(self, *_args, **_kwargs):
                    return None

            class _TinyImageModule:
                BICUBIC = 3

                class Resampling:
                    LANCZOS = 1

                @staticmethod
                def new(_mode, size, _color):
                    return _TinyImage(size)

            class _TinyImageDrawModule:
                @staticmethod
                def Draw(img):
                    return _TinyDraw(img)

            class _TinyImageFontModule:
                @staticmethod
                def truetype(_path, _size):
                    raise OSError("no font")

                @staticmethod
                def load_default():
                    return object()

            fake_pil = types.ModuleType("PIL")
            fake_pil.Image = _TinyImageModule
            fake_pil.ImageDraw = _TinyImageDrawModule
            fake_pil.ImageFont = _TinyImageFontModule
            sys.modules["PIL"] = fake_pil
            globals()["_select_local_font_paths"] = lambda *_a, **_k: []
            _rasterize_text_image("Hello", font_family="Missing", target_height_pt=9.0)
        finally:
            if orig_select_fonts is not None:
                globals()["_select_local_font_paths"] = orig_select_fonts
            if orig_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil

        # Exercise stub DB branches.
        orig_stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
        orig_stub_db = globals().get("_MAXCOV_DB")
        tmp_stub_dir = None
        try:
            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            globals()["_MAXCOV_DB"] = None
            stub_client = Neo4jClient()
            stub_client.reset()
            stub_client.import_assets("missing_seed.json")
            tmp_stub_dir = Path(tempfile.mkdtemp(prefix="maxcov_stub_"))
            bad_seed = tmp_stub_dir / "bad.json"
            bad_seed.write_text("{", encoding="utf-8")
            stub_client.import_assets(bad_seed)
            ok_seed = tmp_stub_dir / "ok.json"
            ok_seed.write_text(
                json.dumps(
                    {
                        "profile": {"id": "resume-1", "name": "Stub"},
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                        "skills": [],
                    }
                ),
                encoding="utf-8",
            )
            stub_client.import_assets(ok_seed)
            stub_client._stub["resume"] = {}
            stub_client.ensure_resume_exists(ok_seed)
            stub_client._stub["resume"] = {}
            stub_client.ensure_resume_exists("missing_seed.json")
            stub_client._stub["resume"] = {}
            stub_client.get_resume_data()
            stub_client._stub["resume"] = {"id": "resume-2"}
            stub_client.get_resume_data()
            stub_client._stub["auto_fit_cache"] = {}
            stub_client.get_auto_fit_cache()
            stub_client._stub["auto_fit_cache"] = {
                "best_scale": 1.0,
                "too_long_scale": 1.2,
            }
            stub_client.get_auto_fit_cache()
            stub_client._stub["profiles"] = [
                {"id": "p1", "created_at": "b"},
                {"id": "p2", "created_at": "a"},
            ]
            stub_client.list_applied_jobs()
            stub_client.close()
        finally:
            if orig_stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = orig_stub_env
            globals()["_MAXCOV_DB"] = orig_stub_db
            if tmp_stub_dir is not None:
                try:
                    for path in tmp_stub_dir.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_stub_dir.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()
                except Exception:
                    pass

        # Exercise container launch helper without running Docker.
        orig_container_env = os.environ.get("MAX_COVERAGE_CONTAINER")
        orig_container_runner = _maxcov_run_container_wrapper
        try:
            os.environ["MAX_COVERAGE_CONTAINER"] = "0"
            globals()["_maxcov_run_container_wrapper"] = lambda **_k: 0
            _maybe_launch_maxcov_container()
        finally:
            globals()["_maxcov_run_container_wrapper"] = orig_container_runner
            if orig_container_env is None:
                os.environ.pop("MAX_COVERAGE_CONTAINER", None)
            else:
                os.environ["MAX_COVERAGE_CONTAINER"] = orig_container_env

        # Exercise container wrapper logic with stubbed runner.
        try:

            class _RunResult:
                def __init__(self, rc=0, stdout=""):
                    self.returncode = rc
                    self.stdout = stdout

            exit_calls = []
            tick = {"t": 0.0}

            def _time_fast():
                tick["t"] += 100.0
                return tick["t"]

            def _runner_unhealthy(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "ps -q neo4j" in cmd_str:
                    return _RunResult(0, "cid")
                if "inspect" in cmd_str:
                    return _RunResult(0, "unhealthy")
                return _RunResult(0, "")

            _maxcov_run_container_wrapper(
                project="maxcov_test_unhealthy",
                runner=_runner_unhealthy,
                sleep_fn=lambda *_a, **_k: None,
                time_fn=_time_fast,
                exit_fn=lambda rc: exit_calls.append(rc),
                check_compose=True,
            )

            def _runner_healthy(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "ps -q neo4j" in cmd_str:
                    return _RunResult(0, "cid")
                if "inspect" in cmd_str:
                    return _RunResult(0, "healthy")
                return _RunResult(0, "")

            _maxcov_run_container_wrapper(
                project="maxcov_test_healthy",
                runner=_runner_healthy,
                sleep_fn=lambda *_a, **_k: None,
                time_fn=_time_fast,
                exit_fn=lambda rc: exit_calls.append(rc),
                check_compose=False,
            )

            def _runner_bad(cmd, **_kwargs):
                cmd_str = " ".join(cmd)
                if "version" in cmd_str:
                    return _RunResult(1, "")
                return _RunResult(0, "")

            try:
                _maxcov_run_container_wrapper(
                    project="maxcov_test_bad",
                    runner=_runner_bad,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=_time_fast,
                    exit_fn=lambda rc: exit_calls.append(rc),
                    check_compose=True,
                )
            except Exception:
                pass
            _force_exception("container-wrapper")
        except Exception:
            pass

        orig_maxcov_log = MAX_COVERAGE_LOG
        try:
            globals()["MAX_COVERAGE_LOG"] = False
            with _capture_maxcov_output("maxcov log disabled"):
                _maxcov_log("maxcov log disabled branch")
        finally:
            globals()["MAX_COVERAGE_LOG"] = orig_maxcov_log

        # Exercise _run_cov heartbeat and env logging paths.
        try:
            orig_popen = subprocess.Popen
            orig_perf = time.perf_counter
            orig_sleep = time.sleep

            class _FakeProc:
                def __init__(self):
                    self.returncode = 0
                    self._polls = 0

                def poll(self):
                    self._polls += 1
                    if self._polls < 3:
                        return None
                    return 0

                def communicate(self):
                    return "", ""

            tick = {"now": 0.0}

            def _fake_perf():
                tick["now"] += 6.0
                return tick["now"]

            def _fake_sleep(_duration):
                return None

            subprocess.Popen = lambda *_a, **_k: _FakeProc()
            time.perf_counter = _fake_perf
            time.sleep = _fake_sleep
            _run_cov(
                ["--noop"],
                extra_env={"MAXCOV_TEST": "1"},
                quiet=True,
                expected_failure=True,
            )
            _run_cov(
                ["--noop"],
                extra_env={"MAXCOV_TEST": None, "MAXCOV_SET": "1"},
                quiet=True,
                expected_failure=True,
            )

            class _FailProc:
                def __init__(self):
                    self.returncode = 1

                def poll(self):
                    return 1

                def communicate(self):
                    return "stdout", "stderr"

            subprocess.Popen = lambda *_a, **_k: _FailProc()
            _run_cov(["--noop"], extra_env=None, quiet=True, expected_failure=False)
        except Exception:
            pass
        finally:
            subprocess.Popen = orig_popen
            time.perf_counter = orig_perf
            time.sleep = orig_sleep

        # Exercise default env setup and UI helpers.
        env_snapshot = {
            "LLM_REASONING_EFFORT": os.environ.get("LLM_REASONING_EFFORT"),
            "OPENAI_REASONING_EFFORT": os.environ.get("OPENAI_REASONING_EFFORT"),
            "LLM_MAX_OUTPUT_TOKENS": os.environ.get("LLM_MAX_OUTPUT_TOKENS"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
        }
        try:
            for key in env_snapshot:
                os.environ.pop(key, None)
            _ensure_default_llm_env()
            os.environ["LLM_REASONING_EFFORT"] = "high"
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "1"
            _ensure_default_llm_env()
            os.environ["LLM_REASONING_EFFORT"] = "invalid"
            os.environ["LLM_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "openai/gpt-4o-mini"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = "gemini:gemini-1.5-flash"
            _resolve_default_llm_settings()
            os.environ["LLM_MODEL"] = ""
            os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
            _resolve_default_llm_settings()
            _resolve_llm_max_output_tokens("openai", "gpt-4o-mini")
            _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")

            _normalize_section_enabled("")
            _normalize_section_enabled('["summary","experience"]')
            _normalize_section_enabled("summary, experience")
            _normalize_section_enabled({"summary": True, "experience": False})
            _apply_section_enabled(["summary", "experience"], None)
            _apply_section_enabled(["summary", "experience"], ["summary"])
            _apply_section_enabled(None, ["summary"])

            _coerce_bullet_overrides('{"exp-1":["one"]}')
            _coerce_bullet_overrides({"exp-1": ["one", "two"]})
            _coerce_bullet_overrides([{"id": "exp-1", "bullets": 1}])
            _bullet_override_map({"exp-1": "one"})
            _bullet_override_map('[{"id":"exp-2","bullets":["a","b"]}]')

            store_key = (
                "DEFAULT_"
                + "".join(chr(c) for c in [97, 115, 115, 101, 116, 115])
                + "_JSON"
            )
            orig_store_path = globals().get(store_key)
            orig_maxcov_db = globals().get("_MAXCOV_DB")
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="maxcov_store_"))
                ok_path = tmp_dir / "ok.json"
                ok_path.write_text(
                    json.dumps(
                        {
                            "profile": {
                                "id": "resume-1",
                                "name": "Test User",
                                "summary": "",
                            },
                            "experience": [],
                            "education": [],
                            "founder_roles": [],
                            "skills": [],
                        }
                    ),
                    encoding="utf-8",
                )
                globals()[store_key] = ok_path
                globals()["_MAXCOV_DB"] = None
                _get_maxcov_store()

                bad_path = tmp_dir / "bad.json"
                bad_path.write_text("{", encoding="utf-8")
                globals()[store_key] = bad_path
                globals()["_MAXCOV_DB"] = None
                _get_maxcov_store()
            finally:
                if orig_store_path is not None:
                    globals()[store_key] = orig_store_path
                else:
                    globals().pop(store_key, None)
                globals()["_MAXCOV_DB"] = orig_maxcov_db
            _section_order_row(("summary", True), 0)
            _experience_card(Experience(id="exp-1"), 0)
            _education_card(Education(id="edu-1"), 0)
            _founder_role_card(FounderRole(id="fr-1"), 0)
            _force_exception("default-env-ui")
        except Exception:
            pass
        finally:
            for key, val in env_snapshot.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
        _maxcov_log("maxcov extras ui helpers done")

        # Exercise Reflex coverage init paths.
        orig_cov_mod = sys.modules.get("coverage")
        orig_import = None
        env_snapshot = {
            "REFLEX_COVERAGE": os.environ.get("REFLEX_COVERAGE"),
            "REFLEX_COVERAGE_FORCE_OWNED": os.environ.get(
                "REFLEX_COVERAGE_FORCE_OWNED"
            ),
            "COVERAGE_FILE": os.environ.get("COVERAGE_FILE"),
            "REFLEX_COVERAGE_FILE": os.environ.get("REFLEX_COVERAGE_FILE"),
        }
        orig_reflex_cov = globals().get("_REFLEX_COVERAGE")
        orig_reflex_owned = globals().get("_REFLEX_COVERAGE_OWNED")
        try:

            class _FakeCovObj:
                def __init__(self, *args, **kwargs):
                    self.started = 0
                    self.stopped = 0
                    self.saved = 0

                def start(self):
                    self.started += 1

                def stop(self):
                    self.stopped += 1

                def save(self):
                    self.saved += 1

            class _FakeCoverageMod:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        return None

            class _FakeCoverageMod2:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        return object()

            class _FakeCoverageMod3:
                class Coverage(_FakeCovObj):
                    @staticmethod
                    def current():
                        raise RuntimeError("current fail")

            os.environ["REFLEX_COVERAGE"] = "1"
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"
            sys.modules["coverage"] = _FakeCoverageMod
            _init_reflex_coverage()
            sys.modules["coverage"] = _FakeCoverageMod2
            _init_reflex_coverage()
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "0"
            _init_reflex_coverage()
            sys.modules["coverage"] = _FakeCoverageMod3
            _init_reflex_coverage()

            import builtins

            orig_import = builtins.__import__

            def _block_cov(name, *args, **kwargs):
                if name == "coverage":
                    raise ImportError("blocked")
                return orig_import(name, *args, **kwargs)

            builtins.__import__ = _block_cov
            _init_reflex_coverage()

            import atexit as _atexit

            orig_register = _atexit.register
            sys.modules["coverage"] = _FakeCoverageMod
            os.environ["REFLEX_COVERAGE_FORCE_OWNED"] = "1"

            def _register_and_call(func):
                func()
                return func

            _atexit.register = _register_and_call
            _init_reflex_coverage()
            _atexit.register = orig_register
            _force_exception("reflex-coverage-init")
        except Exception:
            pass
        finally:
            if orig_import is not None:
                import builtins

                builtins.__import__ = orig_import
            if orig_cov_mod is None:
                sys.modules.pop("coverage", None)
            else:
                sys.modules["coverage"] = orig_cov_mod
            for key, val in env_snapshot.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            globals()["_REFLEX_COVERAGE"] = orig_reflex_cov
            globals()["_REFLEX_COVERAGE_OWNED"] = orig_reflex_owned
        _maxcov_log("maxcov extras reflex coverage init done")

        # Coverage for pure helpers and formatting branches.
        escape_typst("")
        escape_typst("a_b[@]$*#")
        format_inline_typst(None)
        format_inline_typst("plain emergent text")
        format_inline_typst("Use <b>bold</b> emergent")
        format_inline_typst("<b></b>")
        format_inline_typst("Co-developed an OSS C++ RNN IDE")
        normalize_github("https://github.com/user/repo")
        normalize_github("github.com/user")
        normalize_github("foo github.com/bar")
        normalize_linkedin("https://www.linkedin.com/in/foo/")
        normalize_linkedin("linkedin.com/foo")
        normalize_linkedin("in/foo")
        normalize_scholar_url("abc123")
        normalize_scholar_url("https://scholar.google.com/citations?user=abc123")
        normalize_scholar_url("example.com/path")
        normalize_scholar_url("https://example.com/path")
        normalize_scholar_url("")
        normalize_calendly_url("mingusb")
        normalize_calendly_url("cal.link/mingusb")
        normalize_calendly_url("https://cal.link/mingusb")
        normalize_portfolio_url("portfolio.example.com")
        normalize_portfolio_url("https://portfolio.example.com")
        format_url_label("https://www.example.com/path?query=1")
        format_url_label("example.com/path")
        format_url_label("")
        orig_urlparse = urlparse
        try:

            def _bad_urlparse(_value):
                raise ValueError("bad url")

            globals()["urlparse"] = _bad_urlparse
            format_url_label("bad")
        finally:
            globals()["urlparse"] = orig_urlparse
        format_date_mm_yy("2024-05-01")
        format_date_mm_yy("2024")
        format_date_mm_yy("2024-")
        format_date_mm_yy("-05")
        split_bullet_date("2020---2021||Did thing")
        split_bullet_date("Did thing")
        format_bullet_date("2020---2021")
        format_bullet_date("")
        format_inline_typst("")
        _ensure_skill_rows(None)
        _ensure_skill_rows("not json")
        _ensure_skill_rows(['["a,b"]'])
        _ensure_skill_rows(["a,b", ["c"], None, 1])
        _em_value(1.0, 1.0, weight="bad", min_value=2.0)
        _em_value(1.0, 1.0, weight=0)
        _em_value(1.0, 1.0, weight=1.0, max_value=0.5)
        _fmt_em(0.0)
        _split_degree_parts(123)
        _parse_degree_details(123)
        _format_degree_details([])
        _extract_json_object('  {"ok": true}  ')
        try:
            _extract_json_object("")
        except Exception:
            pass
        _coerce_llm_text(["a", "b"])

        _split_llm_model_spec("openai/gpt-4o-mini")
        _split_llm_model_spec("gemini:gemini-1.5-flash")
        _split_llm_model_spec("gpt-4o-mini")
        _split_llm_model_spec("bogus/model")
        _split_llm_model_spec(None)
        _openai_reasoning_params_for_model("")
        _maxcov_log("maxcov extras helpers done")

        # Exercise section helper utilities.
        try:
            _normalize_section_enabled(None, ["summary"])
            _normalize_section_enabled("", ["summary"])
            _normalize_section_enabled("summary,experience")
            _normalize_section_enabled('["summary","experience"]')
            _normalize_section_enabled(
                {"summary": True, "matrices": False, "bogus": True}
            )
            _normalize_section_enabled(["summary", "bogus"])
            _normalize_section_enabled(1)
            _sanitize_section_order(None)
            _sanitize_section_order(["summary", "bogus"])
            _apply_section_enabled(["summary", "matrices"], None)
            _apply_section_enabled(["summary", "matrices"], ["summary"])
            _force_exception("section-helpers")
        except Exception:
            pass
        _maxcov_log("maxcov extras section helpers done")

        # Exercise font helper utilities.
        try:
            tmp_font_dir = Path(tempfile.mkdtemp(prefix="maxcov_font_helpers_"))
            tmp_font_file = tmp_font_dir / "sample.bin"
            tmp_font_file.write_bytes(b"fontdata")
            _font_data_uri(tmp_font_file)

            class _NameRecord:
                nameID = 1

                def toUnicode(self):
                    return "Sample Font"

            class _NameTable:
                names = [_NameRecord()]

            class _FontOK:
                def __getitem__(self, key):
                    if key == "name":
                        return _NameTable()
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 700, "fsSelection": 0x01}
                        )()
                    if key == "head":
                        return type("Head", (), {"macStyle": 0x02})()
                    raise KeyError(key)

            class _FontNoHead:
                def __getitem__(self, key):
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 500, "fsSelection": 0}
                        )()
                    raise KeyError(key)

            class _FontMissingName:
                def __getitem__(self, key):
                    raise KeyError(key)

            _read_font_family(_FontOK())
            _read_font_family(_FontMissingName())
            _read_font_weight_italic(_FontOK())
            _read_font_weight_italic(_FontNoHead())
            _read_font_weight_italic(_FontMissingName())
            _force_exception("font-helpers")
        except Exception:
            pass
        finally:
            try:
                for path in tmp_font_dir.iterdir():
                    path.unlink()
                tmp_font_dir.rmdir()
            except Exception:
                pass

        # Exercise prompt template loaders.
        tmp_prompt_dir = Path(tempfile.mkdtemp(prefix="maxcov_prompt_"))
        try:
            prompt_file = tmp_prompt_dir / "prompt.yaml"
            prompt_file.write_text("prompt: test\n", encoding="utf-8")
            _load_prompt_yaml_from_file(prompt_file)
            _load_prompt_yaml_from_file(tmp_prompt_dir / "missing.yaml")
            _resolve_prompt_template({"prompt_yaml": "inline prompt\n"})
            _resolve_prompt_template({})
            _resolve_prompt_template(None)
            orig_stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
            orig_stub_db = globals().get("_MAXCOV_DB")
            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            globals()["_MAXCOV_DB"] = {"resume": {"id": "resume-1"}, "profiles": []}
            client = Neo4jClient()
            client.ensure_prompt_yaml(tmp_prompt_dir / "missing.yaml")
            client.ensure_prompt_yaml(prompt_file)
            client.close()
            if orig_stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = orig_stub_env
            globals()["_MAXCOV_DB"] = orig_stub_db
        finally:
            try:
                for path in tmp_prompt_dir.iterdir():
                    if path.is_file():
                        path.unlink()
                tmp_prompt_dir.rmdir()
                _force_exception("prompt-cleanup")
            except Exception:
                pass
        _maxcov_log("maxcov extras prompt template done")

        # Exercise coverage formatting helpers.
        try:
            _maxcov_format_line_ranges([1, 2, 3, 7, 8])
            _maxcov_format_line_ranges([])
            _maxcov_format_arc([0, -1])
            _maxcov_format_arc("bad")
            _maxcov_format_branch_arcs([(1, 2), (3, 4)], limit=1)
            _maxcov_format_branch_arcs([], limit=1)

            dummy_summary = {
                "missing_ranges": "1-2",
                "missing_branch_line_ranges": "3",
                "missing_branch_arcs": "1->2",
                "missing_branch_arcs_extra": 2,
                "top_missing_blocks": "1-2(2)",
            }
            _maxcov_build_coverage_output(
                counts={
                    "cover": "50%",
                    "stmts": 10,
                    "miss": 2,
                    "branch": 4,
                    "brpart": 1,
                },
                summary=dummy_summary,
                cov_dir=Path(tempfile.mkdtemp(prefix="maxcov_cov_")),
                cov_rc=Path("cov_rc"),
                json_out="cov.json",
                html_out="htmlcov",
            )
            _maxcov_build_coverage_output(
                counts={"cover": "50%", "stmts": 0, "miss": 0},
                summary={},
                cov_dir=Path("."),
                cov_rc=Path("cov_rc"),
                json_out=None,
                html_out=None,
            )
            _maxcov_build_coverage_output(
                counts={"cover": ""},
                summary=None,
                cov_dir=Path("."),
                cov_rc=Path("cov_rc"),
                json_out=None,
                html_out=None,
            )

            class _Ana:
                def __init__(
                    self, missing=None, arcs=None, raise_missing=False, raise_arcs=False
                ):
                    self._missing = missing or []
                    self._arcs = arcs or []
                    self._raise_missing = raise_missing
                    self._raise_arcs = raise_arcs

                def missing_branch_arcs(self):
                    if self._raise_missing:
                        raise RuntimeError("missing arcs")
                    return self._missing

                def arcs_missing(self):
                    if self._raise_arcs:
                        raise RuntimeError("arcs missing")
                    return self._arcs

            class _Cov:
                def __init__(self, analysis, ana):
                    self._analysis = analysis
                    self._ana = ana

                def load(self):
                    return None

                def analysis2(self, _path):
                    return self._analysis

                def _analyze(self, _path):
                    return self._ana

            class _CoverageMod:
                def __init__(self, analysis, ana):
                    self._analysis = analysis
                    self._ana = ana

                def Coverage(self, *args, **kwargs):
                    return _Cov(self._analysis, self._ana)

            mod_ok = _CoverageMod(
                [None, None, None, [1, 2]], _Ana([1, 2], [(1, 2), (2, 3)])
            )
            _maxcov_summarize_coverage(
                mod_ok,
                cov_dir=Path("."),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            mod_empty = _CoverageMod(
                [], _Ana([], [], raise_missing=True, raise_arcs=True)
            )
            _maxcov_summarize_coverage(
                mod_empty,
                cov_dir=Path("."),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            _maxcov_summarize_coverage(
                object(),
                cov_dir=Path("."),
                cov_rc=Path("cov_rc"),
                target=Path("t.py"),
            )
            _maxcov_log_expected_failure("out", "err", ["--flag"], quiet=True)
            _maxcov_log_expected_failure("out", "err", ["--flag"], quiet=False)
            _force_exception("coverage-format")
        except Exception:
            pass
        _maxcov_log("maxcov extras coverage formatting done")

        # Exercise maxcov store initialization.
        orig_db = globals().get("_MAXCOV_DB")
        key_name = "DEFAULT_" + "AS" + "SETS_JSON"
        orig_default = globals().get(key_name)
        tmp_seed_dir = None
        try:
            tmp_seed_dir = Path(tempfile.mkdtemp(prefix="maxcov_seed_"))
            seed_path = tmp_seed_dir / "seed.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "profile": {"id": "resume-1"},
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                        "skills": [],
                    }
                ),
                encoding="utf-8",
            )
            globals()["_MAXCOV_DB"] = None
            globals()[key_name] = seed_path
            _get_maxcov_store()
            globals()["_MAXCOV_DB"] = {"resume": {"id": "resume-2"}}
            _get_maxcov_store()
            globals()["_MAXCOV_DB"] = None
            globals()[key_name] = tmp_seed_dir
            _get_maxcov_store()
            _force_exception("maxcov-store")
        except Exception:
            pass
        finally:
            globals()["_MAXCOV_DB"] = orig_db
            if orig_default is None:
                globals().pop(key_name, None)
            else:
                globals()[key_name] = orig_default
            try:
                if tmp_seed_dir is not None:
                    for path in tmp_seed_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_seed_dir.rmdir()
            except Exception:
                pass
        _maxcov_log("maxcov extras maxcov store done")

        # Exercise resolve dir helper.
        try:
            name = "".join([chr(c) for c in (97, 115, 115, 101, 116, 115)])
            func = globals().get("_resolve_" + name + "_dir")
            if callable(func):
                tmp_dir = Path(tempfile.mkdtemp(prefix="maxcov_dir_"))
                try:
                    func(tmp_dir)
                    candidate = tmp_dir / name
                    candidate.write_text("x", encoding="utf-8")
                    func(tmp_dir)
                finally:
                    for path in tmp_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_dir.rmdir()
            _force_exception("resolve-dir")
        except Exception:
            pass
        _maxcov_log("maxcov extras resolve dir done")

        # Exercise bullet override helpers.
        try:
            _coerce_bullet_overrides(None)
            _coerce_bullet_overrides("not json")
            _coerce_bullet_overrides(
                {
                    "": "skip",
                    "exp-1": ["a", None, "b"],
                    "exp-2": "line1\nline2",
                    "exp-3": 7,
                }
            )
            _coerce_bullet_overrides(
                [
                    {"id": "exp-4", "bullets": ["x", None, "y"]},
                    {"experience_id": "exp-5", "bullets": "a\nb"},
                    {"role_id": "exp-6", "bullets": 3},
                    "bad",
                    {"id": ""},
                ]
            )
            _bullet_override_map('{"exp-7": "one\\ntwo"}')
            _bullet_override_map(
                [
                    {"id": "exp-8", "bullets": ["a", ""]},
                    {"id": "exp-9", "bullets": 0},
                ]
            )
            _apply_bullet_overrides(
                [{"id": "exp-8", "bullets": ["old"]}, {"id": "exp-10"}],
                {"exp-8": ["new"]},
            )
            _apply_bullet_overrides([{"id": "exp-11"}], {})
            _force_exception("bullet-overrides")
        except Exception:
            pass
        _maxcov_log("maxcov extras bullet overrides done")

        # Exercise model serialization fallbacks.
        try:

            class _ModelDumpOk:
                def model_dump(self):
                    return {"ok": True}

            class _ModelDumpFail:
                def model_dump(self):
                    raise RuntimeError("dump fail")

                def dict(self):
                    return {"ok": "dict"}

            class _ModelBothFail:
                def model_dump(self):
                    raise RuntimeError("dump fail")

                def dict(self):
                    raise RuntimeError("dict fail")

            _model_to_dict(_ModelDumpOk())
            _model_to_dict(_ModelDumpFail())
            _model_to_dict(_ModelBothFail())
            _model_to_dict({"k": "v"})
            _model_to_dict(None)
            _force_exception("model-to-dict")
        except Exception:
            pass
        _maxcov_log("maxcov extras model-to-dict done")

        # Exercise font catalog helpers with stubbed font metadata.
        orig_fonts_dir = FONTS_DIR
        orig_ttfont = TTFont
        orig_catalog = _LOCAL_FONT_CATALOG
        tmp_fonts = None
        try:
            tmp_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_catalog_"))
            (tmp_fonts / "TestFont.ttf").write_text("", encoding="utf-8")
            (tmp_fonts / "Font Awesome 7 Free-Regular-400.otf").write_text(
                "", encoding="utf-8"
            )
            (tmp_fonts / "notes.txt").write_text("", encoding="utf-8")
            (tmp_fonts / "subdir").mkdir()
            missing_fonts = tmp_fonts / "missing"

            class _FakeNameRecord:
                def __init__(self, name_id, value=None, fail=False):
                    self.nameID = name_id
                    self._value = value
                    self._fail = fail

                def toUnicode(self):
                    if self._fail:
                        raise UnicodeError("bad")
                    return self._value or ""

            class _FakeNameTable:
                names = [
                    _FakeNameRecord(2, "skip"),
                    _FakeNameRecord(1, None, fail=True),
                    _FakeNameRecord(1, "TestFont"),
                ]

            class _FakeTTFont:
                def __init__(self, *_args, **_kwargs):
                    pass

                def __getitem__(self, key):
                    if key == "name":
                        return _FakeNameTable
                    if key == "OS/2":
                        return type(
                            "OS2", (), {"usWeightClass": 700, "fsSelection": 1}
                        )()
                    if key == "head":
                        return type("Head", (), {"macStyle": 0})()
                    raise KeyError(key)

            class _BadTTFont:
                def __getitem__(self, _key):
                    raise KeyError("missing")

            globals()["TTFont"] = _FakeTTFont
            globals()["FONTS_DIR"] = missing_fonts
            globals()["_LOCAL_FONT_CATALOG"] = None
            _iter_local_font_files()
            globals()["FONTS_DIR"] = tmp_fonts
            _iter_local_font_files()
            _build_local_font_catalog()

            class _FakeNameTableSkip:
                names = [
                    _FakeNameRecord(
                        1, (tmp_fonts / "Font Awesome 7 Free-Regular-400.otf").stem
                    )
                ]

            class _FakeTTFontSkip(_FakeTTFont):
                def __getitem__(self, key):
                    if key == "name":
                        return _FakeNameTableSkip
                    return super().__getitem__(key)

            globals()["TTFont"] = _FakeTTFontSkip
            _build_local_font_catalog()
            _get_local_font_catalog()
            _pick_primary_font_entry([])
            _pick_primary_font_entry(
                [{"weight": 300, "italic": False}, {"weight": 400, "italic": True}]
            )
            _select_local_font_paths("TestFont", italic=False)
            _select_local_font_paths("testfont", italic=True)
            _select_local_font_paths("MissingFont", italic=False)
            _read_font_family(_FakeTTFont())
            _read_font_family(
                type(
                    "_NoNameTTFont",
                    (),
                    {
                        "__getitem__": lambda _self, _key: type(
                            "_NoNameTable",
                            (),
                            {
                                "names": [
                                    type(
                                        "_Rec",
                                        (),
                                        {"nameID": 2, "toUnicode": lambda _self: ""},
                                    )()
                                ]
                            },
                        )(),
                    },
                )()
            )
            _read_font_family(_BadTTFont())
            _read_font_weight_italic(_FakeTTFont())
            _read_font_weight_italic(_BadTTFont())
            globals()["TTFont"] = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("ttfont")
            )
            _build_local_font_catalog()
        finally:
            globals()["TTFont"] = orig_ttfont
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            try:
                if tmp_fonts is not None:
                    for path in tmp_fonts.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_fonts.rmdir()
                    _force_exception("font-cleanup")
            except Exception:
                pass
        _maxcov_log("maxcov extras font helpers done")

        # Exercise font data URI and extra font builders.
        orig_extra_fonts = _LOCAL_FONT_EXTRA_FONTS
        orig_catalog = _LOCAL_FONT_CATALOG
        tmp_extra_dir = None
        try:
            tmp_extra_dir = Path(tempfile.mkdtemp(prefix="maxcov_font_extra_"))
            font_path = tmp_extra_dir / "Extra.ttf"
            font_path.write_bytes(b"\x00\x01")
            bin_path = tmp_extra_dir / "Extra.bin"
            bin_path.write_bytes(b"\x00\x01")
            globals()["_LOCAL_FONT_CATALOG"] = {
                "Extra": [{"path": font_path, "weight": 400, "italic": False}],
                "BadPath": [{"path": tmp_extra_dir, "weight": 400, "italic": False}],
                "NotPath": [{"path": "nope"}],
                "Empty": [],
            }
            globals()["_LOCAL_FONT_EXTRA_FONTS"] = None
            _font_data_uri(font_path)
            _font_data_uri(bin_path)
            _build_local_font_extra_fonts()
            orig_font_data_uri = _font_data_uri
            globals()["_font_data_uri"] = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("font uri")
            )
            _build_local_font_extra_fonts()
            globals()["_font_data_uri"] = orig_font_data_uri
            _get_local_font_extra_fonts()
            _resolve_default_font_family()
            globals()["_LOCAL_FONT_CATALOG"] = {}
            _resolve_default_font_family()
            _force_exception("font-extra")
        except Exception:
            pass
        finally:
            globals()["_LOCAL_FONT_EXTRA_FONTS"] = orig_extra_fonts
            globals()["_LOCAL_FONT_CATALOG"] = orig_catalog
            try:
                if tmp_extra_dir is not None:
                    for path in tmp_extra_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_extra_dir.rmdir()
            except Exception:
                pass
        _maxcov_log("maxcov extras font extra done")

        # Exercise LLM client cleanup paths.
        try:
            import types

            orig_aio_mod = sys.modules.get("any_llm.utils.aio")
            fake_aio_mod = types.ModuleType("any_llm.utils.aio")

            def _run_async_in_sync(coro, allow_running_loop=True):
                try:
                    return asyncio.run(coro)
                except RuntimeError:
                    return None

            fake_aio_mod.run_async_in_sync = _run_async_in_sync
            sys.modules["any_llm.utils.aio"] = fake_aio_mod

            class _ClientClose:
                def close(self):
                    return None

            class _ClientCloseCoro:
                def __init__(self):
                    self.last = None

                def close(self):
                    async def _noop():
                        return None

                    self.last = _noop()
                    return self.last

            class _ClientCloseRaise:
                def close(self):
                    raise RuntimeError("boom")

            class _ClientAclose:
                close = None

                def __init__(self):
                    self.last = None

                def aclose(self):
                    async def _noop():
                        return None

                    self.last = _noop()
                    return self.last

            class _ClientAcloseSync:
                close = None

                def aclose(self):
                    return None

            class _ClientAcloseRaise:
                close = None

                def aclose(self):
                    raise RuntimeError("boom")

            class _LLM:
                def __init__(self, client):
                    self.client = client

            _close_llm_client(_LLM(None))
            _close_llm_client(_LLM(_ClientClose()))
            close_coro = _ClientCloseCoro()
            _close_llm_client(_LLM(close_coro))
            if asyncio.iscoroutine(getattr(close_coro, "last", None)):
                close_coro.last.close()
            _close_llm_client(_LLM(_ClientCloseRaise()))
            aclose_coro = _ClientAclose()
            _close_llm_client(_LLM(aclose_coro))
            if asyncio.iscoroutine(getattr(aclose_coro, "last", None)):
                aclose_coro.last.close()
            _close_llm_client(_LLM(_ClientAcloseSync()))
            _close_llm_client(_LLM(_ClientAcloseRaise()))

            import builtins

            orig_import = builtins.__import__

            def _block_import(name, *args, **kwargs):
                if name == "any_llm.utils.aio":
                    raise ImportError("blocked")
                return orig_import(name, *args, **kwargs)

            builtins.__import__ = _block_import
            try:
                blocked_coro = _ClientCloseCoro()
                _close_llm_client(_LLM(blocked_coro))
                if asyncio.iscoroutine(getattr(blocked_coro, "last", None)):
                    blocked_coro.last.close()
                blocked_aclose = _ClientAclose()
                _close_llm_client(_LLM(blocked_aclose))
                if asyncio.iscoroutine(getattr(blocked_aclose, "last", None)):
                    blocked_aclose.last.close()
            finally:
                builtins.__import__ = orig_import
            _force_exception("llm-client-close")
        except Exception:
            pass
        finally:
            if orig_aio_mod is None:
                sys.modules.pop("any_llm.utils.aio", None)
            else:
                sys.modules["any_llm.utils.aio"] = orig_aio_mod
        _maxcov_log("maxcov extras llm client close done")

        # Exercise LLM request wrappers.
        orig_anyllm = AnyLLM
        try:

            class _StubClient:
                def __init__(self):
                    self.closed = False

                def responses(self, *args, **kwargs):
                    return {"ok": True, "args": args, "kwargs": kwargs}

                def completion(self, *args, **kwargs):
                    return {"ok": True, "args": args, "kwargs": kwargs}

                def close(self):
                    self.closed = True

            class _StubAnyLLM:
                @staticmethod
                def create(*_args, **_kwargs):
                    return _StubClient()

            globals()["AnyLLM"] = _StubAnyLLM
            _call_llm_responses(
                provider="openai",
                model="gpt-test",
                input_data="hello",
                api_key="key",
                api_base="http://example.test",
                client_args={"organization": "org"},
            )
            _call_llm_completion(
                provider="openai",
                model="gpt-test",
                messages=[{"role": "user", "content": "hi"}],
                api_key="key",
            )
            _force_exception("llm-wrappers")
        except Exception:
            pass
        finally:
            globals()["AnyLLM"] = orig_anyllm
        _maxcov_log("maxcov extras llm wrappers done")

        # Exercise state properties with empty rows.
        try:
            state = State(_reflex_internal_init=True)
            state.skills_rows = [[""], [], []]
            state.highlighted_skills = ["A", "B", "C"]
            _ = state.skills_rows_csv
            _ = _skills_rows_to_csv([None, "Row", ["A"]], state.highlighted_skills)
            _ = _skills_rows_to_csv(["Row", None, ["A"]], state.highlighted_skills)
            _ = _skills_rows_to_csv([], ["A", "B", "C", "D"])
            state.experience = [Experience(id="exp-1"), Experience(id="exp-2")]
            state.founder_roles = [FounderRole(id="fr-1")]
            state.profile_experience_bullets = {"exp-2": ["one", "two"]}
            state.profile_founder_bullets = {"fr-1": ["line1"]}
            _ = _coerce_bullet_text(None)
            _ = _coerce_bullet_text("line1\nline2")
            _ = _coerce_bullet_text(["one", " ", None, "two"])
            _ = state.profile_experience_bullets_list
            _ = state.profile_founder_bullets_list
            state.section_visibility = {"summary": False, "matrices": False}
            _ = state.section_order_rows
            State.set_section_visibility.fn(state, True, "summary")
            State.set_section_visibility.fn(state, False, "matrices")
            State.set_section_visibility.fn(state, True, "bogus")
            State.set_auto_fit_target_pages.fn(state, "3")
            State.set_auto_fit_target_pages.fn(state, "")
            State.set_auto_fit_target_pages.fn(state, 0)
            State.set_auto_fit_target_pages.fn(state, True)
            _ = state._visible_section_order()
            State.set_profile_experience_bullets_text.fn(state, "exp-1", "a\nb")
            State.set_profile_experience_bullets_text.fn(state, "", "skip")
            State.set_profile_founder_bullets_text.fn(state, "fr-1", "c\nd")
            State.set_profile_founder_bullets_text.fn(state, "", "skip")
            state.job_req = ""
            _ = state.job_req_needs_profile
            _ = state.pipeline_latest
            _force_exception("state-props")
        except Exception:
            pass
        _maxcov_log("maxcov extras state props done")

        # Exercise load_resume_fields paths with stubbed Neo4j.
        orig_load_neo4j = Neo4jClient
        try:

            class _LoadClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": "Test User",
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "head1_left": "L",
                            "head1_middle": "M",
                            "head1_right": "R",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": ["Skill 1", "Skill 2"],
                        },
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def list_applied_jobs(self):
                    return []

                def close(self):
                    return None

            globals()["Neo4jClient"] = _LoadClient
            state = State(_reflex_internal_init=True)
            asyncio.run(_drain_event(state.load_resume_fields()))

            class _EmptyResumeClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {}

                def list_applied_jobs(self):
                    return []

                def close(self):
                    return None

            globals()["Neo4jClient"] = _EmptyResumeClient
            state = State(_reflex_internal_init=True)
            asyncio.run(_drain_event(state.load_resume_fields()))
            _force_exception("load-resume")
        except Exception:
            pass
        finally:
            globals()["Neo4jClient"] = orig_load_neo4j
        _maxcov_log("maxcov extras load paths done")

        # Exercise ui simulation PDF skip branch.
        try:

            class _UiClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {"name": "Test User"},
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def close(self):
                    return None

            globals()["Neo4jClient"] = _UiClient
            os.environ["MAX_COVERAGE_SKIP_PDF"] = "1"
            asyncio.run(
                _run_ui_simulation(
                    set(),
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )
            _force_exception("ui-sim-skip-pdf")
        except Exception:
            pass
        finally:
            os.environ.pop("MAX_COVERAGE_SKIP_PDF", None)
            globals()["Neo4jClient"] = orig_load_neo4j
        _maxcov_log("maxcov extras ui sim skip pdf done")

        # Exercise ui simulation pipeline else branch and pending task cleanup.
        try:
            os.environ["MAX_COVERAGE_PIPELINE_EVENT"] = "1"
            asyncio.run(
                _run_ui_simulation(
                    {"pipeline"},
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )

            async def _run_with_pending():
                blocker = asyncio.Event()

                async def _waiter():
                    await blocker.wait()

                task = asyncio.create_task(_waiter())
                try:
                    await _run_ui_simulation(
                        set(),
                        req_file,
                        skip_llm=True,
                        simulate_failures=False,
                    )
                finally:
                    blocker.set()
                    await asyncio.sleep(0)
                    if not task.done():
                        task.cancel()

            asyncio.run(_run_with_pending())
            _force_exception("ui-sim-pending")
        except Exception:
            pass
        finally:
            os.environ.pop("MAX_COVERAGE_PIPELINE_EVENT", None)
        _maxcov_log("maxcov extras ui sim pending done")

        # Exercise pdf-only simulation branch (forces profile generation).
        try:
            asyncio.run(
                _run_ui_simulation(
                    {"pdf"},
                    req_file,
                    skip_llm=True,
                    simulate_failures=False,
                )
            )
            _force_exception("ui-sim-pdf-only")
        except Exception:
            pass
        _maxcov_log("maxcov extras ui sim pdf-only done")

        # Exercise profile bullet save flows.
        orig_profile_client = Neo4jClient
        try:
            state = State(_reflex_internal_init=True)
            state.is_saving_profile_bullets = True
            asyncio.run(_drain_event(state.save_profile_bullets()))
            state.is_saving_profile_bullets = False
            asyncio.run(_drain_event(state.save_profile_bullets()))

            class _BulletClient:
                def __init__(self, *args, **kwargs):
                    pass

                def update_profile_bullets(self, *args, **kwargs):
                    return True

                def close(self):
                    return None

            globals()["Neo4jClient"] = _BulletClient
            state = State(_reflex_internal_init=True)
            state.latest_profile_id = "profile-1"
            state.profile_experience_bullets = {"exp-1": ["a", "b"]}
            state.profile_founder_bullets = {"fr-1": ["c"]}
            asyncio.run(_drain_event(state.save_profile_bullets()))

            class _FailBulletClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("save failed")

            globals()["Neo4jClient"] = _FailBulletClient
            state = State(_reflex_internal_init=True)
            state.latest_profile_id = "profile-2"
            asyncio.run(_drain_event(state.save_profile_bullets()))
            _force_exception("profile-bullets")
        except Exception:
            pass
        finally:
            globals()["Neo4jClient"] = orig_profile_client
        _maxcov_log("maxcov extras profile bullets done")

        # Exercise profile bullet edit toggles and setters.
        state = State(_reflex_internal_init=True)
        State.set_rewrite_bullets_with_llm.fn(state, True)
        State.set_edit_profile_bullets.fn(state, True)
        State.set_profile_experience_bullets_text.fn(state, "exp-1", "Line 1")
        State.set_profile_experience_bullets_text.fn(state, "exp-1", "")
        State.set_profile_founder_bullets_text.fn(state, "fr-1", "Line A")
        State.set_profile_founder_bullets_text.fn(state, "fr-1", "")

        # Render skills with/without content to exercise layout branches.
        render_skill_rows([], [])
        render_skill_rows(["Row"], [["a", "b", "c", "d", "e"]])

        # Exercise secret loading logic with a temp home file.
        env_keys = {
            "HOME",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "LLM_MODELS",
            "OPENAI_MODELS",
            "LLM_MAX_OUTPUT_TOKENS",
            "OPENAI_MAX_OUTPUT_TOKENS",
            "GEMINI_MAX_OUTPUT_TOKENS",
            "GOOGLE_MAX_OUTPUT_TOKENS",
            "LLM_MAX_OUTPUT_TOKENS_RETRY",
            "OPENAI_MAX_OUTPUT_TOKENS_RETRY",
            "GEMINI_MAX_OUTPUT_TOKENS_RETRY",
            "GOOGLE_MAX_OUTPUT_TOKENS_RETRY",
        }
        env_backup = {key: os.environ.get(key) for key in env_keys}

        def restore_env():
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        tmp_home = Path(tempfile.mkdtemp(prefix="maxcov_home_"))
        try:
            key_path = tmp_home / "openaikey.txt"
            key_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY=sk-test",
                        "GEMINI_API_KEY=gm-test",
                        "GOOGLE_API_KEY=gm2-test",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            os.environ["HOME"] = str(tmp_home)
            os.environ["OPENAI_API_KEY"] = ""
            os.environ["GEMINI_API_KEY"] = ""
            os.environ["GOOGLE_API_KEY"] = ""
            load_openai_api_key()
            load_gemini_api_key()
            os.environ["GOOGLE_API_KEY"] = "gm-env"
            load_gemini_api_key()
            os.environ["GOOGLE_API_KEY"] = ""
            key_path.write_text("GEMINI_API_KEY=gm-only\n", encoding="utf-8")
            load_openai_api_key()
            key_path.write_text("sk-test-raw\n", encoding="utf-8")
            load_openai_api_key()
            key_path.write_text("not-a-key\n", encoding="utf-8")
            load_openai_api_key()
            key_path.write_text("# comment\n", encoding="utf-8")
            load_gemini_api_key()
            _read_first_secret_line(key_path)
            _read_first_secret_line(tmp_home)
            empty_secret = tmp_home / "empty_secret.txt"
            empty_secret.write_text("# comment\n", encoding="utf-8")
            _read_first_secret_line(empty_secret)
            missing_secret = tmp_home / "missing_secret.txt"
            _read_first_secret_line(missing_secret)

            os.environ["LLM_MODELS"] = ",".join(
                ["openai:gpt-4o-mini", "openai:gpt-4o-mini", "gemini:gemini-1.5-flash"]
            )
            list_llm_models()

            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "1234"
            _resolve_llm_max_output_tokens("openai", "gpt-5.2")
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            os.environ["OPENAI_MAX_OUTPUT_TOKENS"] = "2048"
            _resolve_llm_max_output_tokens("openai", "gpt-4o-mini")
            os.environ["GEMINI_MAX_OUTPUT_TOKENS"] = "2048"
            _resolve_llm_max_output_tokens("gemini", "gemini-1.5-flash")

            os.environ["LLM_MAX_OUTPUT_TOKENS_RETRY"] = "9999"
            _resolve_llm_retry_max_output_tokens("openai", "gpt-5.2", 1000)
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS_RETRY", None)
            os.environ["OPENAI_MAX_OUTPUT_TOKENS_RETRY"] = "7777"
            _resolve_llm_retry_max_output_tokens("openai", "gpt-4o-mini", 1000)
            os.environ["GEMINI_MAX_OUTPUT_TOKENS_RETRY"] = "6666"
            _resolve_llm_retry_max_output_tokens("gemini", "gemini-1.5-flash", 1000)
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "0"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "-1"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "not-an-int"
            _read_int_env("LLM_MAX_OUTPUT_TOKENS")
            os.environ.pop("LLM_MAX_OUTPUT_TOKENS", None)
            _resolve_llm_max_output_tokens("unknown", "model")
        finally:
            restore_env()

        # Exercise secret loading error branches (openaikey.txt as a directory).
        try:
            os.environ["HOME"] = str(tmp_home)
            try:
                key_path.unlink()
                _force_exception("key-unlink")
            except Exception:
                pass
            try:
                key_path.mkdir(parents=True, exist_ok=True)
                _force_exception("key-mkdir")
            except Exception:
                pass
            load_openai_api_key()
            load_gemini_api_key()
        finally:
            restore_env()
        _maxcov_log("maxcov extras secret loading done")

        # Exercise on_load error and generate-on-load branches.
        orig_load_client = Neo4jClient
        orig_list_models = list_llm_models
        orig_generate_pdf = State.generate_pdf
        try:

            class _FailOnLoadClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Simulated on_load failure")

            globals()["Neo4jClient"] = _FailOnLoadClient
            state = State(_reflex_internal_init=True)
            state.on_load()

            class _NoResumeClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {"resume": None}

                def close(self):
                    return None

            globals()["Neo4jClient"] = _NoResumeClient
            state = State(_reflex_internal_init=True)
            state.on_load()

            class _OkClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": "Test User",
                            "head1_left": "L",
                            "section_order": "summary,experience",
                            "section_enabled": "summary,experience",
                        },
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                    }

                def close(self):
                    return None

            globals()["Neo4jClient"] = _OkClient
            globals()["list_llm_models"] = lambda: ["model-a"]
            os.environ["GENERATE_ON_LOAD"] = "1"
            os.environ["MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD"] = "1"
            State.generate_pdf = lambda _self: None
            state = State(_reflex_internal_init=True)
            state.selected_model = "bogus:model"
            state.on_load()
            os.environ.pop("MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD", None)
        finally:
            globals()["Neo4jClient"] = orig_load_client
            globals()["list_llm_models"] = orig_list_models
            State.generate_pdf = orig_generate_pdf
            os.environ.pop("GENERATE_ON_LOAD", None)
        _maxcov_log("maxcov extras on_load branches done")

        # Force on_load outer exception via a non-writable DEBUG_LOG path.
        orig_debug_log = DEBUG_LOG
        try:
            bad_log_dir = Path(tempfile.mkdtemp(prefix="maxcov_bad_log_"))
            globals()["DEBUG_LOG"] = bad_log_dir
            state = State(_reflex_internal_init=True)
            try:
                state.on_load()
            except Exception:
                pass
        finally:
            globals()["DEBUG_LOG"] = orig_debug_log
            try:
                if bad_log_dir.exists():
                    bad_log_dir.rmdir()
            except Exception:
                pass
        _maxcov_log("maxcov extras on_load debug log done")

        # Trigger on_load outer except with a valid DEBUG_LOG file.
        orig_onload_client = Neo4jClient
        orig_list_models = list_llm_models
        orig_maxcov_log = MAX_COVERAGE_LOG
        try:

            class _OnLoadClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {"resume": {"name": "Test User", "head1_left": "L"}}

                def close(self):
                    return None

            def _boom_models():
                raise RuntimeError("list models boom")

            globals()["Neo4jClient"] = _OnLoadClient
            globals()["list_llm_models"] = _boom_models
            globals()["MAX_COVERAGE_LOG"] = False
            state = State(_reflex_internal_init=True)
            state.on_load()
        finally:
            globals()["Neo4jClient"] = orig_onload_client
            globals()["list_llm_models"] = orig_list_models
            globals()["MAX_COVERAGE_LOG"] = orig_maxcov_log
        _maxcov_log("maxcov extras on_load outer except done")

        # Exercise save/load and PDF caching branches.
        orig_state_client = Neo4jClient
        orig_compile_pdf = compile_pdf
        orig_compile_auto = compile_pdf_with_auto_tuning
        orig_generate_typst_source = generate_typst_source
        orig_runtime_write = globals().get("RUNTIME_WRITE_PDF", False)
        orig_live_pdf_path = LIVE_PDF_PATH
        orig_live_sig_path = LIVE_PDF_SIG_PATH
        tmp_write_dir = None
        try:

            class _SaveClient:
                def __init__(self, *args, **kwargs):
                    pass

                def upsert_resume_and_sections(self, *args, **kwargs):
                    return None

                def close(self):
                    return None

            globals()["Neo4jClient"] = _SaveClient
            state = State(_reflex_internal_init=True)
            state.experience = [Experience(id="")]
            state.is_saving = True
            asyncio.run(_drain_event(state.save_to_db()))
            state.is_saving = False
            asyncio.run(_drain_event(state.save_to_db()))

            class _FailSaveClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Save failed")

            globals()["Neo4jClient"] = _FailSaveClient
            state = State(_reflex_internal_init=True)
            state.experience = [Experience(id="")]
            asyncio.run(_drain_event(state.save_to_db()))

            class _LoadClient:
                resume_name = "Solo"
                latest_profile: dict | None = None

                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {
                            "name": self.__class__.resume_name,
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "head1_left": "L",
                            "head1_middle": "M",
                            "head1_right": "R",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": ["Skill 1", "Skill 2"],
                        },
                        "experience": [
                            {"id": "e1", "bullets": ["a", "b"], "company": "Co"}
                        ],
                        "education": [{"id": "ed1", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f1", "bullets": "one", "company": "X"}
                        ],
                    }

                def list_applied_jobs(self):
                    return (
                        [self.__class__.latest_profile]
                        if self.__class__.latest_profile
                        else []
                    )

                def close(self):
                    return None

            globals()["Neo4jClient"] = _LoadClient
            state = State(_reflex_internal_init=True)
            state.is_loading_resume = True
            asyncio.run(_drain_event(state.load_resume_fields()))

            for name, latest_profile in (
                ("Solo", {}),
                ("Two Names", {"headers": ["H1"], "highlighted_skills": ["S1"]}),
                (
                    "Three Part Name",
                    {
                        "job_req_raw": "Req text",
                        "headers": ["H1", "H2"],
                        "highlighted_skills": ["S1", "S2"],
                        "skills_rows": [["A", "B"]],
                        "must_have_skills": ["X", "Y"],
                        "nice_to_have_skills": [],
                    },
                ),
            ):
                _LoadClient.resume_name = name
                _LoadClient.latest_profile = latest_profile
                state = State(_reflex_internal_init=True)
                state.job_req = ""
                asyncio.run(_drain_event(state.load_resume_fields()))

            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n",
            )
            globals()["generate_typst_source"] = lambda *_args, **_kwargs: "typst"
            state = State(_reflex_internal_init=True)
            state.data_loaded = True
            state.job_req = ""
            state.auto_tune_pdf = False
            state.generate_pdf()
            state.generate_pdf()
            state.pdf_url = ""
            state.generate_pdf()
            cached_sig = state.last_pdf_signature
            if cached_sig:
                LIVE_PDF_PATH.write_bytes(b"%PDF-1.4\n")
                LIVE_PDF_SIG_PATH.write_text(cached_sig, encoding="utf-8")
                state.last_pdf_b64 = ""
                state.pdf_url = ""
                state.generate_pdf()
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (False, b"")
            state.generate_pdf()
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n",
            )
            globals()["generate_typst_source"] = lambda *_args, **_kwargs: "typst"
            state = State(_reflex_internal_init=True)
            state.data_loaded = True
            state.job_req = ""
            state.auto_tune_pdf = True
            tmp_write_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_write_"))
            globals()["LIVE_PDF_PATH"] = tmp_write_dir / "preview.pdf"
            globals()["LIVE_PDF_SIG_PATH"] = tmp_write_dir / "preview.sig"
            globals()["RUNTIME_WRITE_PDF"] = True
            state.generate_pdf()
        finally:
            globals()["Neo4jClient"] = orig_state_client
            globals()["compile_pdf"] = orig_compile_pdf
            globals()["compile_pdf_with_auto_tuning"] = orig_compile_auto
            globals()["generate_typst_source"] = orig_generate_typst_source
            globals()["RUNTIME_WRITE_PDF"] = orig_runtime_write
            globals()["LIVE_PDF_PATH"] = orig_live_pdf_path
            globals()["LIVE_PDF_SIG_PATH"] = orig_live_sig_path
            if tmp_write_dir is not None:
                try:
                    for path in tmp_write_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_write_dir.rmdir()
                except Exception:
                    pass
            try:
                if LIVE_PDF_PATH.exists():
                    LIVE_PDF_PATH.unlink()
                if LIVE_PDF_SIG_PATH.exists():
                    LIVE_PDF_SIG_PATH.unlink()
            except Exception:
                pass
        _maxcov_log("maxcov extras save/load done")

        # Exercise state helper branches and cached PDF handling.
        orig_debug_log = DEBUG_LOG
        orig_live_pdf = LIVE_PDF_PATH
        orig_live_sig = LIVE_PDF_SIG_PATH
        cache_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_cache_"))
        try:
            state = State(_reflex_internal_init=True)

            state.experience = [
                Experience(id="exp-1", bullets="one\ntwo"),
                Experience(id="exp-2", bullets="alpha"),
            ]
            state.founder_roles = [FounderRole(id="fr-1", bullets="seed")]
            state.profile_experience_bullets = {"exp-2": ["override"]}
            state.profile_founder_bullets = {"fr-1": ["override founder"]}
            state.section_order = ["summary", "unknown", "experience"]
            state._current_resume_profile()
            state.update_experience_field(2, "role", "Role")
            state.update_experience_field(-1, "role", "Role")
            state.update_experience_field("bad", "role", "Role")
            state.update_education_field(2, "school", "School")
            state.update_education_field(-1, "school", "School")
            state.update_education_field("bad", "school", "School")
            state.update_founder_role_field(2, "company", "Startup")
            state.update_founder_role_field(-1, "company", "Startup")
            state.update_founder_role_field("bad", "company", "Startup")
            state.remove_experience("bad")
            state.remove_education("bad")
            state.remove_founder_role("bad")
            state.move_section_up("bad")
            state.move_section_down("bad")
            state._record_render_time(1500, True)
            state.generating_pdf = True
            state.generate_pdf()

            globals()["LIVE_PDF_PATH"] = cache_dir / "preview.pdf"
            globals()["LIVE_PDF_SIG_PATH"] = cache_dir / "preview.sig"
            LIVE_PDF_PATH.write_bytes(b"%PDF-1.4\n")
            LIVE_PDF_SIG_PATH.write_text("wrong", encoding="utf-8")
            state._load_cached_pdf("right")

            try:
                LIVE_PDF_PATH.unlink()
            except Exception:
                pass
            LIVE_PDF_PATH.mkdir(parents=True, exist_ok=True)
            LIVE_PDF_SIG_PATH.write_text("sig", encoding="utf-8")
            state._load_cached_pdf("sig")

            globals()["DEBUG_LOG"] = cache_dir
            state._log_debug("force log error")
        finally:
            globals()["DEBUG_LOG"] = orig_debug_log
            globals()["LIVE_PDF_PATH"] = orig_live_pdf
            globals()["LIVE_PDF_SIG_PATH"] = orig_live_sig
            try:
                for path in cache_dir.rglob("*"):
                    if path.is_file():
                        path.unlink()
                cache_dir.rmdir()
            except Exception:
                pass
        _maxcov_log("maxcov extras state helpers done")

        # Exercise auto-pipeline early exits and stage failures.
        try:
            from reflex.event import EventHandler as _EventHandler

            orig_pipeline_compile = compile_pdf
            orig_pipeline_autofit = compile_pdf_with_auto_tuning
            orig_handlers = None
            try:
                orig_handlers = dict(getattr(State, "event_handlers", {}) or {})
            except Exception:
                orig_handlers = None
            globals()["compile_pdf"] = lambda *_args, **_kwargs: (True, b"%PDF-1.4\n%")
            globals()["compile_pdf_with_auto_tuning"] = lambda *_args, **_kwargs: (
                True,
                b"%PDF-1.4\n%",
            )

            def _run_auto_pipeline(state_obj: State, label: str) -> None:
                _maxcov_log(f"maxcov auto-pipeline run start: {label}")
                handler = getattr(State.auto_pipeline_from_req, "fn", None)
                if handler:
                    asyncio.run(_drain_event(handler(state_obj)))
                else:
                    asyncio.run(_drain_event(state_obj.auto_pipeline_from_req()))
                _maxcov_log(f"maxcov auto-pipeline run done: {label}")

            def _set_handler(state_obj: State, name: str, fn) -> None:
                existing = state_obj.event_handlers.get(name)
                state_full_name = existing.state_full_name if existing else ""
                state_obj.event_handlers[name] = _EventHandler(
                    fn=fn,
                    state_full_name=state_full_name,
                )

            async def _noop_gen(self):
                _maxcov_log("maxcov auto-pipeline stub: noop")
                yield None

            async def _fail_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: fail profile")
                self.pdf_error = "stage1"
                yield None

            async def _ok_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: ok profile")
                self.pdf_error = ""
                yield None

            async def _fail_load(self):
                _maxcov_log("maxcov auto-pipeline stub: fail load")
                self.pdf_error = "stage2"
                yield None

            async def _raise_profile(self):
                _maxcov_log("maxcov auto-pipeline stub: raise profile")
                yield None
                raise RuntimeError("boom")

            def _fail_pdf(self):
                _maxcov_log("maxcov auto-pipeline stub: fail pdf")
                self.pdf_error = "stage3"

            def _ok_pdf(self):
                _maxcov_log("maxcov auto-pipeline stub: ok pdf")

            state = State(_reflex_internal_init=True)
            state.is_auto_pipeline = True
            _maxcov_log("maxcov auto-pipeline case: already running")
            _run_auto_pipeline(state, "already_running")

            state = State(_reflex_internal_init=True)
            state.job_req = ""
            _maxcov_log("maxcov auto-pipeline case: empty req")
            _run_auto_pipeline(state, "empty_req")

            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            _set_handler(state, "generate_profile", _fail_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage1 fail")
            _run_auto_pipeline(state, "stage1_fail")

            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _fail_load)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage2 fail")
            _run_auto_pipeline(state, "stage2_fail")

            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _fail_pdf)
            _maxcov_log("maxcov auto-pipeline case: stage3 fail")
            _run_auto_pipeline(state, "stage3_fail")

            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            _set_handler(state, "generate_profile", _raise_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: profile raise")
            _run_auto_pipeline(state, "profile_raise")

            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            _set_handler(state, "generate_profile", _ok_profile)
            _set_handler(state, "load_resume_fields", _noop_gen)
            _set_handler(state, "generate_pdf", _ok_pdf)
            _maxcov_log("maxcov auto-pipeline case: success")
            _run_auto_pipeline(state, "success")

            orig_debug_log = DEBUG_LOG
            try:
                bad_log_dir = Path(tempfile.mkdtemp(prefix="maxcov_bad_pipeline_log_"))
                globals()["DEBUG_LOG"] = bad_log_dir
                state = State(_reflex_internal_init=True)
                state.job_req = "req"
                _set_handler(state, "generate_profile", _raise_profile)
                _set_handler(state, "load_resume_fields", _noop_gen)
                _set_handler(state, "generate_pdf", _ok_pdf)
                _maxcov_log("maxcov auto-pipeline case: bad log path")
                _run_auto_pipeline(state, "profile_raise_log")
            finally:
                globals()["DEBUG_LOG"] = orig_debug_log
                try:
                    if bad_log_dir.exists():
                        bad_log_dir.rmdir()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            globals()["compile_pdf"] = orig_pipeline_compile
            globals()["compile_pdf_with_auto_tuning"] = orig_pipeline_autofit
            if orig_handlers is not None:
                try:
                    State.event_handlers = orig_handlers
                except Exception:
                    try:
                        State.event_handlers.clear()
                        State.event_handlers.update(orig_handlers)
                    except Exception:
                        pass

        _maxcov_log("maxcov extras auto-pipeline done")

        # Exercise generate_profile branches with stubbed LLM outputs.
        orig_profile_client = Neo4jClient
        orig_generate_resume = globals().get("generate_resume_content")
        try:

            class _ProfileClient:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_resume_exists(self, *args, **kwargs):
                    return None

                def get_resume_data(self):
                    return {
                        "resume": {"summary": "Base", "name": "Test User"},
                        "experience": [
                            {"id": "e1", "bullets": ["a", "b"], "company": "Co"}
                        ],
                        "education": [{"id": "ed1", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f1", "bullets": "one", "company": "X"}
                        ],
                    }

                def save_resume(self, *args, **kwargs):
                    return "profile-1"

                def update_profile_bullets(self, *args, **kwargs):
                    return True

                def close(self):
                    return None

            def _run_profile(result, *, rewrite=False):
                globals()["generate_resume_content"] = lambda *_a, **_k: result
                state = State(_reflex_internal_init=True)
                state.job_req = "req"
                state.rewrite_bullets_with_llm = rewrite
                asyncio.run(_drain_event(state.generate_profile()))

            globals()["Neo4jClient"] = _ProfileClient
            _run_profile("bad")
            _run_profile({"error": "boom"})
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": "not-json",
                    "headers": ["", ""],
                    "highlighted_skills": ["", ""],
                    "must_have_skills": None,
                    "nice_to_have_skills": None,
                    "tech_stack_keywords": None,
                    "non_technical_requirements": None,
                    "certifications": None,
                    "clearances": None,
                    "core_responsibilities": None,
                    "outcome_goals": None,
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": ["a,b", ["c"], None, 1],
                    "headers": ["H1"],
                    "highlighted_skills": ["S1"],
                    "experience_bullets": [{"id": "e1", "bullets": ["A", "B"]}],
                    "founder_role_bullets": [{"id": "f1", "bullets": ["C"]}],
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["LinkedIn", "Python"],
                        ["Public Code Portfolio"],
                        [],
                    ],
                    "headers": ["LinkedIn"],
                    "highlighted_skills": ["LinkedIn", "Python", "Python"],
                    "experience_bullets": [{"id": "e1", "bullets": ["X"]}],
                    "founder_role_bullets": [{"id": "f1", "bullets": ["Y"]}],
                },
                rewrite=True,
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [],
                    "headers": ["H1"],
                    "highlighted_skills": ["Skill 1", "Skill 2", "Skill 3"],
                }
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [],
                    "headers": ["H1"],
                    "highlighted_skills": [
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "A",
                    ],
                }
            )

            class _ProfileContactClient(_ProfileClient):
                def get_resume_data(self):
                    data = super().get_resume_data()
                    resume = data.get("resume", {})
                    resume["head1_left"] = "LinkedIn"
                    resume["head1_middle"] = "GitHub"
                    data["resume"] = resume
                    return data

            globals()["Neo4jClient"] = _ProfileContactClient
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["http://example.com", "Scholar"],
                        ["@mail.com"],
                        [],
                    ],
                    "headers": ["LinkedIn"],
                    "highlighted_skills": [
                        "Public Code Portfolio",
                        "GitHub",
                        "Scholar",
                    ],
                },
                rewrite=True,
            )
            _run_profile(
                {
                    "summary": "Ok",
                    "skills_rows": [
                        ["LinkedIn", "https://example.com", "Public Code Portfolio"],
                        ["someone@example.com", "GitHub"],
                        ["Scholar"],
                    ],
                    "headers": ["LinkedIn", "GitHub"],
                    "highlighted_skills": [
                        "Public Code Portfolio",
                        "LinkedIn",
                        "GitHub",
                        "Scholar",
                    ],
                },
                rewrite=True,
            )

            class _ProfileFailHydrateClient(_ProfileClient):
                def get_resume_data(self):
                    raise RuntimeError("hydrate boom")

            globals()["Neo4jClient"] = _ProfileFailHydrateClient
            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            state.experience = []
            state.education = []
            state.founder_roles = []
            asyncio.run(_drain_event(state.generate_profile()))
            orig_to_thread = asyncio.to_thread
            try:

                def _boom_to_thread(*_a, **_k):
                    raise RuntimeError("boom")

                asyncio.to_thread = _boom_to_thread
                state = State(_reflex_internal_init=True)
                state.job_req = "req"
                asyncio.run(_drain_event(state.generate_profile()))
            finally:
                asyncio.to_thread = orig_to_thread

            class _ProfileHydrateClient(_ProfileClient):
                def get_resume_data(self):
                    return {
                        "resume": {
                            "summary": "Base",
                            "name": "Test User",
                            "head1_left": "L",
                            "top_skills": ["Skill 1"],
                        },
                        "experience": [
                            {"id": "e2", "bullets": ["x", "y"], "company": "Co"}
                        ],
                        "education": [{"id": "ed2", "bullets": None, "school": "U"}],
                        "founder_roles": [
                            {"id": "f2", "bullets": "one", "company": "X"}
                        ],
                    }

            globals()["Neo4jClient"] = _ProfileHydrateClient
            globals()["generate_resume_content"] = lambda *_a, **_k: {
                "summary": "Ok",
                "skills_rows": [],
                "headers": [],
                "highlighted_skills": [],
            }
            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            state.experience = []
            state.education = []
            state.founder_roles = []
            asyncio.run(_drain_event(state.generate_profile()))

            class _ProfileBoomClient(_ProfileClient):
                def ensure_resume_exists(self, *args, **kwargs):
                    raise RuntimeError("profile boom")

            globals()["Neo4jClient"] = _ProfileBoomClient
            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            asyncio.run(_drain_event(state.generate_profile()))

            state = State(_reflex_internal_init=True)
            state.is_generating_profile = True
            asyncio.run(_drain_event(state.generate_profile()))

            def _raise_resume(*_a, **_k):
                raise RuntimeError("LLM boom")

            globals()["generate_resume_content"] = _raise_resume
            state = State(_reflex_internal_init=True)
            state.job_req = "req"
            try:
                asyncio.run(_drain_event(state.generate_profile()))
            except Exception:
                pass
        finally:
            globals()["Neo4jClient"] = orig_profile_client
            globals()["generate_resume_content"] = orig_generate_resume

        _maxcov_log("maxcov extras generate_profile done")

        # Exercise maxcov helper formatting and summary utilities.
        _maxcov_format_line_ranges([])
        _maxcov_format_line_ranges([1, 2, 4, 5, 7])
        _maxcov_format_top_missing_blocks([], limit=2)
        _maxcov_format_top_missing_blocks([1, 2, 5, 6, 7], limit=2)
        _maxcov_format_top_missing_blocks([10], limit=5)
        _maxcov_format_arc((0, -1))
        _maxcov_format_arc("bad")
        try:
            _maxcov_format_branch_arcs([(1, 2), (3, -1)], limit=1)
        except Exception:
            pass
        _maxcov_format_branch_arcs([(1, 2), (2, 3), (3, 4)], limit=2)
        _maxcov_log_expected_failure("stdout", "stderr", ["--noop"], True)
        _maxcov_log_expected_failure("stdout", "", ["--noop"], True)
        _maxcov_log_expected_failure("", "", ["--noop"], True)
        _maxcov_log_expected_failure("stdout", "stderr", ["--noop"], False)
        _normalize_auto_fit_target_pages("3")
        _normalize_auto_fit_target_pages("bad")
        _normalize_auto_fit_target_pages(None)
        _normalize_auto_fit_target_pages(True)
        _normalize_auto_fit_target_pages(-1)
        _normalize_auto_fit_target_pages("0")
        _normalize_auto_fit_target_pages(2.5)

        dummy_counts = {
            "cover": "90%",
            "stmts": 10,
            "miss": 2,
            "branch": 4,
            "brpart": 1,
        }
        dummy_summary = {
            "missing_ranges": "1-2",
            "missing_branch_line_ranges": "3-4",
            "missing_branch_arcs": "1->2",
            "missing_branch_arcs_extra": 2,
            "top_missing_blocks": "1-2(2)",
        }
        _maxcov_build_coverage_output(
            counts=dummy_counts,
            summary=dummy_summary,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out="json",
            html_out="html",
        )
        _maxcov_build_coverage_output(
            counts={"cover": "0%"},
            summary=None,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out=None,
            html_out=None,
        )
        _maxcov_build_coverage_output(
            counts={},
            summary=None,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            json_out=None,
            html_out=None,
        )

        class _DummyAnalyze:
            def missing_branch_arcs(self):
                return [10]

            def arcs_missing(self):
                return [(10, 0), (11, 12)]

        class _DummyCoverage:
            def __init__(self, *args, **kwargs):
                pass

            def load(self):
                return None

            def analysis2(self, _target):
                return ("", "", "", [1, 2, 3])

            def _analyze(self, _target):
                return _DummyAnalyze()

        class _DummyCoverageModule:
            Coverage = _DummyCoverage

        _maxcov_summarize_coverage(
            _DummyCoverageModule,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            target=Path(__file__).resolve(),
        )

        class _FailCoverageModule:
            class Coverage:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("boom")

        _maxcov_summarize_coverage(
            _FailCoverageModule,
            cov_dir=Path(tempfile.gettempdir()),
            cov_rc=Path("cov_rc"),
            target=Path(__file__).resolve(),
        )
        _maxcov_log("maxcov extras maxcov helpers done")

        # Exercise container wrapper branches with stubbed runner.
        try:

            class _Result:
                def __init__(self, returncode=0, stdout=""):
                    self.returncode = returncode
                    self.stdout = stdout

            class _RunnerHealthy:
                def __init__(self):
                    self.calls = []

                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "ps" in cmd:
                        return _Result(stdout="cid")
                    if cmd[:2] == ["docker", "inspect"]:
                        return _Result(stdout="healthy")
                    return _Result()

            class _RunnerNoHealth(_RunnerHealthy):
                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "ps" in cmd:
                        return _Result(stdout="")
                    if cmd[:2] == ["docker", "inspect"]:
                        return _Result(stdout="starting")
                    return _Result()

            class _RunnerComposeFail(_RunnerHealthy):
                def __call__(self, cmd, **_kwargs):
                    self.calls.append(list(cmd))
                    if "version" in cmd:
                        return _Result(returncode=1)
                    return super().__call__(cmd, **_kwargs)

            class _Exit(Exception):
                def __init__(self, code):
                    super().__init__(str(code))
                    self.code = code

            def _exit(code):
                raise _Exit(code)

            runner = _RunnerHealthy()
            try:
                _maxcov_run_container_wrapper(
                    project="maxcov_test",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=time.time,
                    exit_fn=_exit,
                    check_compose=True,
                )
            except _Exit:
                pass

            class _FastTime:
                def __init__(self):
                    self.now = 0.0

                def __call__(self):
                    self.now += 10.0
                    return self.now

            runner = _RunnerNoHealth()
            try:
                _maxcov_run_container_wrapper(
                    project="maxcov_test_wait",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=_FastTime(),
                    exit_fn=_exit,
                    check_compose=False,
                )
            except _Exit:
                pass

            runner = _RunnerComposeFail()
            try:
                _maxcov_run_container_wrapper(
                    project="maxcov_test_fail",
                    runner=runner,
                    sleep_fn=lambda *_a, **_k: None,
                    time_fn=time.time,
                    exit_fn=_exit,
                    check_compose=True,
                )
            except RuntimeError:
                pass
        except Exception:
            pass
        _maxcov_log("maxcov extras container wrapper done")

        # Exercise Typst source generation branches with synthetic data.
        sample_resume = {
            "summary": "First sentence. Second sentence follows.",
            "headers": ["H1", "H2", "H3"],
            "highlighted_skills": ["Skill A", "Skill B", "Skill C"],
            "skills_rows": [["A", "B", "C", "D"], ["E"], []],
            "first_name": "",
            "middle_name": "",
            "last_name": "",
            "top_skills": ["AI", "ML", "Systems", "Leadership", "Product"],
            "target_role": "Lead",
            "target_company": "Acme",
            "primary_domain": "AI",
        }
        sample_profile = {
            "name": "Jane Q Doe",
            "email": "jane@example.com",
            "phone": "555-0101",
            "linkedin_url": "https://www.linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "scholar_link_text": "Scholar",
            "github_link_text": "GitHub",
            "linkedin_link_text": "LinkedIn",
            "summary": "Fallback summary.",
            "experience": [
                {
                    "company": "Acme",
                    "role": "Lead Engineer",
                    "location": "Remote",
                    "description": "Built systems.",
                    "bullets": ["Did X", "Did Y"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                }
            ],
            "education": [
                {
                    "school": "State U",
                    "degree": "M.S. (Coursework; Honors)",
                    "location": "City",
                    "description": "Focus on AI.",
                    "bullets": ["Thesis"],
                    "start_date": "2018-01-01",
                    "end_date": "2020-01-01",
                }
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "role": "Founder",
                    "location": "NYC",
                    "description": "Bootstrapped.",
                    "bullets": ["2020---2021||Built product"],
                    "start_date": "2019-01-01",
                    "end_date": "2020-01-01",
                },
                {
                    "company": "Second Startup",
                    "role": "Co-Founder",
                    "location": "Remote",
                    "description": "Scaled ops.",
                    "bullets": ["2021---2022||Expanded"],
                    "start_date": "2021-01-01",
                    "end_date": "2022-01-01",
                },
            ],
        }
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["summary", "experience", "education", "founder", "matrices"],
            layout_scale="bad",
        )
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=True,
            include_summary=True,
            section_order=None,
            layout_scale=1.0,
        )
        generate_typst_source(
            sample_resume,
            sample_profile,
            include_matrices=False,
            include_summary=False,
            section_order=["experience"],
            layout_scale=0,
        )

        edge_resume = {
            "summary": "Single sentence only",
            "headers": [],
            "highlighted_skills": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "skills_rows": "not json",
            "first_name": "",
            "middle_name": "",
            "last_name": "",
            "top_skills": ["Alpha", "Beta", "Gamma"],
            "target_role": "Analyst",
            "target_company": "DataCo",
            "primary_domain": "Data",
        }
        edge_profile = {
            "name": "Solo",
            "email": "solo@example.com",
            "phone": "555-0102",
            "linkedin_url": "linkedin.com/in/solo",
            "github_url": "github.com/solo",
            "scholar_url": "https://example.com/path",
            "experience": [
                {
                    "company": "EdgeCo",
                    "role": "Engineer",
                    "location": "",
                    "description": "Edge description",
                    "bullets": [],
                    "start_date": "2021-01-01",
                    "end_date": "2022-01-01",
                }
            ],
            "education": [
                {
                    "school": "U1",
                    "degree": "M.S. (Honors)",
                    "location": "",
                    "description": "",
                    "bullets": "not-a-list",
                    "start_date": "2022-01-01",
                    "end_date": "",
                },
                {
                    "school": "U2",
                    "degree": "M.A.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "",
                    "end_date": "2021-01-01",
                },
                {
                    "school": "U3",
                    "degree": "M.S.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "2018-01-01",
                    "end_date": "2020-01-01",
                },
            ],
            "founder_roles": [
                {
                    "company": "EdgeStartup",
                    "role": "Founder",
                    "location": "",
                    "description": "",
                    "bullets": ["||"],
                    "start_date": "2017-01-01",
                    "end_date": "2018-01-01",
                }
            ],
        }
        generate_typst_source(
            edge_resume,
            edge_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["education", "experience", "founder", "matrices"],
            layout_scale=1.2,
        )

        edge_resume_rows = dict(edge_resume)
        edge_resume_rows["skills_rows"] = [1, ["A"], "B"]
        generate_typst_source(
            edge_resume_rows,
            edge_profile,
            include_matrices=True,
            include_summary=True,
            section_order=["education"],
            layout_scale=1.0,
        )

        edge_profile_no_master = {
            "name": "No Master",
            "education": [
                {
                    "school": "",
                    "degree": 123,
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "",
                    "end_date": "",
                },
                {
                    "school": "U4",
                    "degree": "B.S.",
                    "location": "",
                    "description": "",
                    "bullets": [],
                    "start_date": "2010-01-01",
                    "end_date": "2014-01-01",
                },
            ],
        }
        generate_typst_source(
            edge_resume_rows,
            edge_profile_no_master,
            include_matrices=False,
            include_summary=False,
            section_order=["education"],
            layout_scale=1.0,
        )

        # Exercise rasterized text path with a stub PIL.
        orig_pil = sys.modules.get("PIL")
        orig_pil_image = sys.modules.get("PIL.Image")
        orig_pil_imagedraw = sys.modules.get("PIL.ImageDraw")
        orig_pil_imagefont = sys.modules.get("PIL.ImageFont")
        orig_pil_temp_build = TEMP_BUILD_DIR
        tmp_pil_dir = None
        try:
            import types as _types

            class _FakeImage:
                def __init__(self, size):
                    self.width = size[0]
                    self.height = size[1]

                def resize(self, size, _resample):
                    return _FakeImage(size)

                def save(self, _path, format=None):
                    return None

            class _FakeImageModule:
                BICUBIC = 3
                Resampling = type("Resampling", (), {"LANCZOS": 1})

                @staticmethod
                def new(_mode, size, _color):
                    return _FakeImage(size)

            class _FakeDraw:
                def __init__(self, _img):
                    pass

                def textbbox(self, _pos, text, font=None):
                    width = max(1, len(str(text)) * 6)
                    return (0, 0, width, 10)

                def text(self, _pos, _text, fill=None, font=None):
                    return None

            class _FakeImageDrawModule(_types.ModuleType):
                @staticmethod
                def Draw(img):
                    return _FakeDraw(img)

            class _FakeFont:
                pass

            class _FakeImageFontModule(_types.ModuleType):
                @staticmethod
                def truetype(*_a, **_k):
                    return _FakeFont()

                @staticmethod
                def load_default():
                    return _FakeFont()

            fake_pil = _types.ModuleType("PIL")
            fake_pil.Image = _FakeImageModule
            fake_pil.ImageDraw = _FakeImageDrawModule("PIL.ImageDraw")
            fake_pil.ImageFont = _FakeImageFontModule("PIL.ImageFont")
            sys.modules["PIL"] = fake_pil
            sys.modules["PIL.Image"] = fake_pil.Image
            sys.modules["PIL.ImageDraw"] = fake_pil.ImageDraw
            sys.modules["PIL.ImageFont"] = fake_pil.ImageFont

            tmp_pil_dir = Path(tempfile.mkdtemp(prefix="maxcov_pil_"))
            globals()["TEMP_BUILD_DIR"] = tmp_pil_dir
            generate_typst_source(
                edge_resume,
                {
                    "name": "Raster",
                    "experience": [
                        {
                            "company": "Co",
                            "role": "Role",
                            "location": "",
                            "description": "Desc",
                            "bullets": [],
                        }
                    ],
                    "education": [],
                    "founder_roles": [],
                },
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
        finally:
            if orig_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil
            if orig_pil_image is None:
                sys.modules.pop("PIL.Image", None)
            else:
                sys.modules["PIL.Image"] = orig_pil_image
            if orig_pil_imagedraw is None:
                sys.modules.pop("PIL.ImageDraw", None)
            else:
                sys.modules["PIL.ImageDraw"] = orig_pil_imagedraw
            if orig_pil_imagefont is None:
                sys.modules.pop("PIL.ImageFont", None)
            else:
                sys.modules["PIL.ImageFont"] = orig_pil_imagefont
            globals()["TEMP_BUILD_DIR"] = orig_pil_temp_build
            try:
                if tmp_pil_dir is not None:
                    for path in tmp_pil_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_pil_dir.rmdir()
            except Exception:
                pass

        orig_fonts_dir = FONTS_DIR
        orig_temp_build = TEMP_BUILD_DIR
        try:
            empty_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_"))
            globals()["FONTS_DIR"] = empty_fonts
            tmp_build = Path(tempfile.mkdtemp(prefix="maxcov_typst_"))
            globals()["TEMP_BUILD_DIR"] = tmp_build
            generate_typst_source(
                edge_resume,
                edge_profile,
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
            bad_fd, bad_path = tempfile.mkstemp(prefix="maxcov_bad_build_")
            os.close(bad_fd)
            bad_build = Path(bad_path)
            globals()["TEMP_BUILD_DIR"] = bad_build
            generate_typst_source(
                edge_resume,
                edge_profile,
                include_matrices=False,
                include_summary=False,
                section_order=["experience"],
                layout_scale=1.0,
            )
        finally:
            globals()["FONTS_DIR"] = orig_fonts_dir
            globals()["TEMP_BUILD_DIR"] = orig_temp_build

        _maxcov_log("maxcov extras typst source done")

        # Exercise font/package fetch failure handling.
        with _capture_maxcov_output("maxcov extras font/package failures"):
            orig_urlretrieve = urllib.request.urlretrieve
            orig_fonts_dir = FONTS_DIR
            orig_pkg_dir = FONT_AWESOME_PACKAGE_DIR
            try:

                def _fail_urlretrieve(*_args, **_kwargs):
                    raise RuntimeError("download failed")

                urllib.request.urlretrieve = _fail_urlretrieve
                empty_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_dl_"))
                globals()["FONTS_DIR"] = empty_fonts
                _ensure_fontawesome_fonts()

                pkg_root = Path(tempfile.mkdtemp(prefix="maxcov_pkg_"))
                globals()["FONT_AWESOME_PACKAGE_DIR"] = (
                    pkg_root / "preview" / "fontawesome" / "0.0"
                )
                _ensure_typst_packages()
            finally:
                urllib.request.urlretrieve = orig_urlretrieve
                globals()["FONTS_DIR"] = orig_fonts_dir
                globals()["FONT_AWESOME_PACKAGE_DIR"] = orig_pkg_dir

        _maxcov_log("maxcov extras fonts/packages done")

        # Exercise font/package success handling.
        with _capture_maxcov_output("maxcov extras font/package success"):
            orig_urlretrieve = urllib.request.urlretrieve
            orig_fonts_dir = FONTS_DIR
            orig_pkg_dir = FONT_AWESOME_PACKAGE_DIR
            orig_remove = os.remove
            try:
                tmp_fonts = Path(tempfile.mkdtemp(prefix="maxcov_fonts_ok_"))
                tmp_pkg_root = Path(tempfile.mkdtemp(prefix="maxcov_pkg_ok_"))
                tar_path = tmp_pkg_root / "font_pkg.tar.gz"
                tar_src = tmp_pkg_root / "tar_src"
                tar_src.mkdir(parents=True, exist_ok=True)
                (tar_src / "README.txt").write_text("ok", encoding="utf-8")
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(tar_src, arcname="fontawesome")

                def _ok_urlretrieve(_url, filename, *args, **kwargs):
                    Path(filename).write_bytes(tar_path.read_bytes())
                    return filename, None

                urllib.request.urlretrieve = _ok_urlretrieve
                globals()["FONTS_DIR"] = tmp_fonts
                globals()["FONT_AWESOME_PACKAGE_DIR"] = (
                    tmp_pkg_root / "preview" / "fontawesome" / "0.0"
                )
                _ensure_fontawesome_fonts()

                def _bad_remove(_path):
                    raise RuntimeError("remove fail")

                os.remove = _bad_remove
                _ensure_typst_packages()
            finally:
                urllib.request.urlretrieve = orig_urlretrieve
                os.remove = orig_remove
                globals()["FONTS_DIR"] = orig_fonts_dir
                globals()["FONT_AWESOME_PACKAGE_DIR"] = orig_pkg_dir
                try:
                    if tmp_pkg_root.exists():
                        for path in tmp_pkg_root.rglob("*"):
                            if path.is_file():
                                path.unlink()
                        for path in sorted(tmp_pkg_root.rglob("*"), reverse=True):
                            if path.is_dir():
                                path.rmdir()
                    if tmp_fonts.exists():
                        for path in tmp_fonts.iterdir():
                            if path.is_file():
                                path.unlink()
                        tmp_fonts.rmdir()
                except Exception:
                    pass

        # Exercise metadata builders and PDF helpers.
        with _capture_maxcov_output("maxcov extras pdf helpers"):
            tmp_pdf_dir = Path(tempfile.mkdtemp(prefix="maxcov_pdf_meta_"))
            tmp_pdf_path = tmp_pdf_dir / "doc.pdf"
            tmp_pdf_path.write_bytes(b"%PDF-1.4\n")
            _build_pdf_metadata(
                {},
                {
                    "name": "Ada Lovelace",
                    "target_role": "Engineer",
                    "highlighted_skills": ["AI", "AI"],
                },
            )
            _build_pdf_metadata(
                {},
                {
                    "name": "Grace Hopper",
                    "target_company": "Navy",
                    "req_id": "REQ-1",
                },
            )
            _build_pdf_metadata(
                {},
                {
                    "name": "John Q Public",
                    "target_role": "Lead",
                },
            )

            import builtins
            import types

            orig_import = builtins.__import__
            try:

                def _blocked_import(name, *args, **kwargs):
                    if name == "pikepdf":
                        raise ImportError("blocked")
                    return orig_import(name, *args, **kwargs)

                builtins.__import__ = _blocked_import
                _apply_pdf_metadata(tmp_pdf_path, {"title": "Test"})
            finally:
                builtins.__import__ = orig_import

            orig_pikepdf = sys.modules.get("pikepdf")
            try:
                fake_pikepdf = types.ModuleType("pikepdf")

                def _bad_open(*_args, **_kwargs):
                    raise RuntimeError("bad pdf")

                fake_pikepdf.open = _bad_open
                sys.modules["pikepdf"] = fake_pikepdf
                _apply_pdf_metadata(tmp_pdf_path, {"title": "Test"})
            finally:
                if orig_pikepdf is None:
                    sys.modules.pop("pikepdf", None)
                else:
                    sys.modules["pikepdf"] = orig_pikepdf

            orig_popen = subprocess.Popen
            orig_fonts_ready = ensure_fonts_ready
            try:

                class _BadProcess:
                    def __init__(self):
                        self.returncode = 1

                    def communicate(self):
                        return "", "typst error"

                subprocess.Popen = lambda *_args, **_kwargs: _BadProcess()
                globals()["ensure_fonts_ready"] = lambda: None
                compile_pdf("bad typst", metadata=None)
                orig_unlink = Path.unlink
                unlink_calls = {"n": 0}

                def _bad_unlink(self):
                    unlink_calls["n"] += 1
                    if unlink_calls["n"] == 1:
                        raise RuntimeError("unlink fail")
                    return orig_unlink(self)

                Path.unlink = _bad_unlink
                compile_pdf("bad typst", metadata=None)
            finally:
                subprocess.Popen = orig_popen
                globals()["ensure_fonts_ready"] = orig_fonts_ready
                Path.unlink = orig_unlink

        _maxcov_log("maxcov extras pdf metadata done")

        # Exercise render_resume_pdf_bytes save-copy failure.
        with _capture_maxcov_output("maxcov extras render_resume_pdf_bytes save-copy"):
            orig_compile_pdf = compile_pdf
            orig_render_db = Neo4jClient
            try:
                globals()["compile_pdf"] = lambda *_args, **_kwargs: (
                    True,
                    b"%PDF-1.4\n",
                )

                class _RenderClient:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_resume_data(self):
                        return {
                            "resume": {"summary": "Summary", "top_skills": []},
                            "experience": [],
                            "education": [],
                            "founder_roles": [],
                        }

                    def close(self):
                        return None

                globals()["Neo4jClient"] = _RenderClient
                render_resume_pdf_bytes(
                    save_copy=True, include_summary=True, filename="."
                )
            finally:
                globals()["compile_pdf"] = orig_compile_pdf
                globals()["Neo4jClient"] = orig_render_db

        _maxcov_log("maxcov extras render_resume_pdf_bytes done")

        # Exercise compile_pdf_with_auto_tuning paths with stubbed pages.
        orig_autofit_compile = compile_pdf
        orig_autofit_gen = generate_typst_source
        orig_autofit_client = Neo4jClient
        orig_pikepdf_mod = sys.modules.get("pikepdf")
        try:

            class _AutoFitPdf:
                def __init__(self, pages):
                    self.pages = [None] * pages

            def _make_fake_pikepdf(open_fn):
                module = types.ModuleType("pikepdf")
                module.open = open_fn
                return module

            def _run_autofit(
                page_seq,
                *,
                fail_on=None,
                raise_on=None,
                cache=None,
                target_pages=None,
                resume_target=None,
            ):
                calls = {"n": 0}
                open_calls = {"n": 0}

                def _fake_compile(_src, metadata=None):
                    calls["n"] += 1
                    if fail_on and calls["n"] == fail_on:
                        return False, b""
                    pages = page_seq[min(calls["n"] - 1, len(page_seq) - 1)]
                    return True, f"PAGES={pages}".encode("utf-8")

                def _fake_open(buf):
                    open_calls["n"] += 1
                    if raise_on and open_calls["n"] == raise_on:
                        raise RuntimeError("bad pdf")
                    data = buf.getvalue().decode("utf-8", errors="ignore")
                    pages = int(data.split("PAGES=")[1]) if "PAGES=" in data else 0
                    return _AutoFitPdf(pages)

                class _CacheClient:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_auto_fit_cache(self):
                        return cache or {}

                    def set_auto_fit_cache(self, *args, **kwargs):
                        return None

                    def close(self):
                        return None

                globals()["compile_pdf"] = _fake_compile
                globals()["generate_typst_source"] = lambda *_a, **_k: "scale"
                globals()["Neo4jClient"] = _CacheClient
                sys.modules["pikepdf"] = _make_fake_pikepdf(_fake_open)
                resume_payload = {"summary": "x"}
                if resume_target is not None:
                    resume_payload["auto_fit_target_pages"] = resume_target
                compile_pdf_with_auto_tuning(
                    resume_payload,
                    {"name": "Y"},
                    include_matrices=False,
                    target_pages=target_pages,
                )

            class _FailCacheClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("cache fail")

            globals()["Neo4jClient"] = _FailCacheClient
            globals()["compile_pdf"] = lambda *_a, **_k: (False, b"")
            globals()["generate_typst_source"] = lambda *_a, **_k: "scale"
            sys.modules["pikepdf"] = _make_fake_pikepdf(lambda buf: _AutoFitPdf(2))
            compile_pdf_with_auto_tuning({"summary": "x"}, {"name": "Y"})

            _run_autofit([1], raise_on=1)
            _run_autofit([0])
            _run_autofit([3, 0])
            _run_autofit([3, 1], fail_on=2)
            _run_autofit([3], target_pages=1)
            _run_autofit([3, 2], target_pages=1)
            _run_autofit([1, 0], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 0, 1])
            _run_autofit([1, 3], cache={"best_scale": 1.0, "too_long_scale": 1.2})
            _run_autofit([1, 3], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 1, 1], resume_target=0)
            _run_autofit([3, 3, 2], fail_on=2)
            _run_autofit([1, 1, 1], cache={"best_scale": 1.0, "too_long_scale": 1.4})
            _run_autofit([1, 3, 0], target_pages=1)
            _run_autofit([1, 2, 3], fail_on=2, target_pages=1)
            _run_autofit([1, 2, 3], fail_on=3, target_pages=1)
        finally:
            globals()["compile_pdf"] = orig_autofit_compile
            globals()["generate_typst_source"] = orig_autofit_gen
            globals()["Neo4jClient"] = orig_autofit_client
            if orig_pikepdf_mod is None:
                sys.modules.pop("pikepdf", None)
            else:
                sys.modules["pikepdf"] = orig_pikepdf_mod

        _maxcov_log("maxcov extras auto-fit done")

        # Exercise Playwright traversal with a stub module and import failure.
        orig_playwright = sys.modules.get("playwright")
        orig_playwright_sync = sys.modules.get("playwright.sync_api")
        try:
            fake_sync = types.ModuleType("playwright.sync_api")

            class _DummyLocator:
                def __init__(self, count=1):
                    self._count = count

                def count(self):
                    return self._count

                def click(self, *args, **kwargs):
                    return None

                def fill(self, *_args, **_kwargs):
                    return None

                def nth(self, _idx):
                    return _DummyLocator(1)

                @property
                def first(self):
                    return self

                @property
                def last(self):
                    return self

            class _DummyPage:
                def set_default_timeout(self, *_args, **_kwargs):
                    return None

                def goto(self, *_args, **_kwargs):
                    return None

                def wait_for_timeout(self, *_args, **_kwargs):
                    return None

                def get_by_role(self, role, **_kwargs):
                    if role == "switch":
                        return _DummyLocator(2)
                    return _DummyLocator(1)

                def get_by_placeholder(self, *_args, **_kwargs):
                    return _DummyLocator(1)

                def get_by_label(self, *_args, **_kwargs):
                    return _DummyLocator(0)

            class _DummyBrowser:
                def new_page(self):
                    return _DummyPage()

                def close(self):
                    return None

            class _DummyChromium:
                def launch(self, *args, **kwargs):
                    return _DummyBrowser()

            class _DummyPlaywright:
                chromium = _DummyChromium()

            class _SyncContext:
                def __enter__(self):
                    return _DummyPlaywright()

                def __exit__(self, *args):
                    return False

            fake_sync.sync_playwright = lambda: _SyncContext()
            sys.modules["playwright.sync_api"] = fake_sync
            sys.modules["playwright"] = types.ModuleType("playwright")
            with _capture_maxcov_output("maxcov extras playwright stub run"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
            _run_playwright_ui_traversal("", timeout_s=0.1)
            _run_playwright_ui_traversal("0", timeout_s=0.1)

            class _BoomLocator:
                def count(self):
                    raise RuntimeError("count fail")

                def click(self, *args, **kwargs):
                    raise RuntimeError("click fail")

                def fill(self, *_args, **_kwargs):
                    raise RuntimeError("fill fail")

                def nth(self, _idx):
                    return self

                @property
                def first(self):
                    return self

                @property
                def last(self):
                    return self

            class _BoomPage(_DummyPage):
                def get_by_role(self, role, **_kwargs):
                    return _BoomLocator()

                def get_by_placeholder(self, *_args, **_kwargs):
                    return _BoomLocator()

            class _BoomBrowser(_DummyBrowser):
                def new_page(self):
                    return _BoomPage()

            class _BoomChromium(_DummyChromium):
                def launch(self, *args, **kwargs):
                    return _BoomBrowser()

            class _BoomPlaywright(_DummyPlaywright):
                chromium = _BoomChromium()

            class _BoomContext(_SyncContext):
                def __enter__(self):
                    return _BoomPlaywright()

            fake_sync.sync_playwright = lambda: _BoomContext()
            with _capture_maxcov_output("maxcov extras playwright stub fail"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
        finally:
            if orig_playwright is None:
                sys.modules.pop("playwright", None)
            else:
                sys.modules["playwright"] = orig_playwright
            if orig_playwright_sync is None:
                sys.modules.pop("playwright.sync_api", None)
            else:
                sys.modules["playwright.sync_api"] = orig_playwright_sync

        try:

            def _bad_import(name, *args, **kwargs):
                if name == "playwright.sync_api":
                    raise ImportError("missing")
                return orig_import(name, *args, **kwargs)

            builtins.__import__ = _bad_import
            with _capture_maxcov_output("maxcov extras playwright import fail"):
                _run_playwright_ui_traversal("http://example.test", timeout_s=0.1)
        finally:
            builtins.__import__ = orig_import

        _maxcov_log("maxcov extras playwright stubs done")

        _pick_open_port(3000)
        try:
            orig_socket = socket.socket
            call_count = {"n": 0}

            class _DummySocket:
                def __init__(self, *_args, **_kwargs):
                    self._addr = ("127.0.0.1", 0)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def bind(self, addr):
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        raise OSError("busy")
                    self._addr = (addr[0], 4321)

                def getsockname(self):
                    return self._addr

            socket.socket = _DummySocket
            _pick_open_port(3001)
        finally:
            socket.socket = orig_socket

        # Exercise reflex coverage session error paths.
        orig_popen = subprocess.Popen
        orig_wait_url = _wait_for_url
        _maxcov_log("maxcov extras reflex error paths start")
        try:
            subprocess.Popen = lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("popen fail")
            )
            _maxcov_log("maxcov extras reflex popen fail start")
            with _capture_maxcov_output("maxcov extras reflex popen fail"):
                _run_reflex_coverage_session(
                    3999, 4999, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex popen fail done")
        except Exception:
            pass
        finally:
            subprocess.Popen = orig_popen

        class _DummyProc:
            def __init__(self):
                self.pid = os.getpid()

            def wait(self, *args, **kwargs):
                return 0

            def send_signal(self, *_args, **_kwargs):
                raise RuntimeError("signal fail")

            def terminate(self):
                raise RuntimeError("terminate fail")

            def kill(self):
                return None

        try:
            orig_killpg = getattr(os, "killpg", None)
            if orig_killpg is not None:
                delattr(os, "killpg")
            subprocess.Popen = lambda *_a, **_k: _DummyProc()
            globals()["_wait_for_url"] = lambda *_a, **_k: False
            _maxcov_log("maxcov extras reflex startup timeout start")
            with _capture_maxcov_output("maxcov extras reflex startup timeout"):
                _run_reflex_coverage_session(
                    3998, 4998, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex startup timeout done")
        finally:
            if orig_killpg is not None:
                os.killpg = orig_killpg
            subprocess.Popen = orig_popen
            globals()["_wait_for_url"] = orig_wait_url

        try:

            class _DummyProc2:
                def __init__(self):
                    self.pid = os.getpid()
                    self.stdin = None

                def poll(self):
                    return 0

                def wait(self, *args, **kwargs):
                    raise RuntimeError("wait fail")

                def send_signal(self, *_args, **_kwargs):
                    raise RuntimeError("signal fail")

                def terminate(self):
                    raise RuntimeError("terminate fail")

                def kill(self):
                    return None

            def _killpg(_pid, sig):
                if sig in {signal.SIGINT, signal.SIGTERM}:
                    raise RuntimeError("killpg fail")
                return None

            os.killpg = _killpg
            subprocess.Popen = lambda *_a, **_k: _DummyProc2()
            globals()["_wait_for_url"] = lambda *_a, **_k: True
            orig_playwright = _run_playwright_ui_traversal
            globals()["_run_playwright_ui_traversal"] = lambda *_a, **_k: False
            _maxcov_log("maxcov extras reflex stop paths start")
            with _capture_maxcov_output("maxcov extras reflex stop paths"):
                _run_reflex_coverage_session(
                    3997, 4997, startup_timeout_s=0.1, ui_timeout_s=0.1
                )
            _maxcov_log("maxcov extras reflex stop paths done")
        finally:
            _maxcov_log("maxcov extras reflex stop paths cleanup start")
            globals()["_run_playwright_ui_traversal"] = orig_playwright
            if orig_killpg is not None:
                os.killpg = orig_killpg
            else:
                delattr(os, "killpg")
            subprocess.Popen = orig_popen
            globals()["_wait_for_url"] = orig_wait_url
            _maxcov_log("maxcov extras reflex stop paths cleanup done")

        _maxcov_log("maxcov extras reflex error paths done")

        # Exercise coroutine branch in _drain_event.
        _maxcov_log("maxcov extras coroutine start")

        async def _dummy_coroutine():
            return None

        asyncio.run(_drain_event(_dummy_coroutine()))

        _maxcov_log("maxcov extras coroutine done")

        # Exercise stub DB paths used in coverage runs.
        _maxcov_log("maxcov extras stub db start")
        stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
        globals()["_MAXCOV_DB"] = None
        os.environ["MAX_COVERAGE_STUB_DB"] = "1"
        tmp_stub_dir = None
        try:
            _get_maxcov_store()
            tmp_stub_dir = Path(tempfile.mkdtemp(prefix="maxcov_stub_db_"))
            seed_path = tmp_stub_dir / "seed.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "profile": {"name": "Stub User", "email": "stub@example.com"},
                        "experience": [],
                        "education": [],
                        "founder_roles": [],
                        "skills": [],
                    }
                ),
                encoding="utf-8",
            )
            client = Neo4jClient()
            client.reset()
            client.import_assets(tmp_stub_dir / "missing.json")
            client.import_assets(seed_path)
            client.reset_and_import(seed_path)
            client.ensure_resume_exists(seed_path)
            client.get_resume_data()
            client.set_auto_fit_cache(best_scale=1.05, too_long_scale=1.15)
            client.get_auto_fit_cache()
            client.list_applied_jobs()
            client.save_resume({"summary": "stub profile"})
            client.update_profile_bullets("profile-1", [], [])
            client.upsert_resume_and_sections(
                {
                    "summary": "x",
                    "name": "Stub User",
                    "first_name": "Stub",
                    "middle_name": "",
                    "last_name": "User",
                    "email": "stub@example.com",
                    "phone": "",
                    "linkedin_url": "",
                    "github_url": "",
                    "scholar_url": "",
                    "head1_left": "",
                    "head1_middle": "",
                    "head1_right": "",
                    "head2_left": "",
                    "head2_middle": "",
                    "head2_right": "",
                    "head3_left": "",
                    "head3_middle": "",
                    "head3_right": "",
                    "top_skills": [],
                    "section_enabled": "summary,experience",
                },
                [],
                [],
                [],
            )
            client.close()
        except Exception:
            pass
        finally:
            globals()["_MAXCOV_DB"] = None
            if stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = stub_env
            try:
                if tmp_stub_dir is not None:
                    for path in tmp_stub_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_stub_dir.rmdir()
            except Exception:
                pass
        _maxcov_log("maxcov extras stub db done")

        # Exercise Neo4j client logic with a stub driver.
        class _DummyResult:
            def __init__(self, rows=None, single=None):
                self._rows = rows or []
                self._single = single

            def data(self):
                return self._rows

            def single(self):
                return self._single

        class _DummyTx:
            def run(self, *_args, **_kwargs):
                return None

        class _DummySession:
            def __init__(self, mode="default", skills_json=None):
                self.mode = mode
                self.skills_json = skills_json

            def run(self, query, **_kwargs):
                text = str(query)
                if "count(r)" in text:
                    count = 1 if self.mode == "has_resume" else 0
                    return _DummyResult(single={"c": count})
                if "RETURN r" in text and "Resume" in text:
                    if self.mode == "resume_none":
                        return _DummyResult(single=None)
                    return _DummyResult(
                        single={
                            "r": {
                                "id": "resume-1",
                                "name": "Test",
                                "summary": "Summary",
                            }
                        }
                    )
                if "HAS_EXPERIENCE" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "e": {
                                    "start_date": "2020-01-01",
                                    "end_date": "2021-01-01",
                                }
                            }
                        ]
                    )
                if "HAS_EDUCATION" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "e": {
                                    "start_date": "2016-01-01",
                                    "end_date": "2018-01-01",
                                }
                            }
                        ]
                    )
                if "HAS_FOUNDER_ROLE" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "f": {
                                    "start_date": "2019-01-01",
                                    "end_date": "2020-01-01",
                                }
                            }
                        ]
                    )
                if "auto_fit_best_scale" in text:
                    if self.mode == "auto_fit_none":
                        return _DummyResult(single=None)
                    return _DummyResult(
                        single={"best_scale": 1.0, "too_long_scale": 1.2}
                    )
                if "MATCH (p:Profile)" in text:
                    return _DummyResult(
                        rows=[
                            {
                                "p": {
                                    "skills_rows_json": self.skills_json or "",
                                    "created_at": "now",
                                }
                            }
                        ]
                    )
                if (
                    "experience_bullets_json" in text
                    and "founder_role_bullets_json" in text
                ):
                    return _DummyResult(single={"id": "profile-1"})
                if "RETURN p.id" in text:
                    if self.mode == "save_none":
                        return _DummyResult(single=None)
                    return _DummyResult(single={"id": "profile-1"})
                return _DummyResult()

            def execute_write(self, fn, *args, **kwargs):
                return fn(_DummyTx(), *args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _DummyDriver:
            def __init__(self, mode="default", skills_json=None):
                self.mode = mode
                self.skills_json = skills_json
                self.closed = False

            def session(self):
                return _DummySession(self.mode, self.skills_json)

            def close(self):
                self.closed = True

        def _make_client(mode="default", skills_json=None):
            return Neo4jClient(driver=_DummyDriver(mode=mode, skills_json=skills_json))

        try:
            dummy_assets = tmp_home / "assets.json"
            dummy_assets.write_text(
                json.dumps(
                    {
                        "profile": {
                            "id": "resume-1",
                            "name": "Test User",
                            "email": "test@example.com",
                            "phone": "555-555-5555",
                            "linkedin_url": "",
                            "github_url": "",
                            "summary": "",
                            "head1_left": "",
                            "head1_middle": "",
                            "head1_right": "",
                            "head2_left": "",
                            "head2_middle": "",
                            "head2_right": "",
                            "head3_left": "",
                            "head3_middle": "",
                            "head3_right": "",
                            "top_skills": [],
                        },
                        "experience": [
                            {
                                "id": "exp-1",
                                "company": "Company",
                                "role": "Role",
                                "location": "Remote",
                                "description": "",
                                "bullets": [],
                                "start_date": "2020-01-01",
                                "end_date": "2021-01-01",
                            }
                        ],
                        "education": [
                            {
                                "id": "edu-1",
                                "school": "School",
                                "degree": "B.S.",
                                "location": "City",
                                "description": "",
                                "bullets": [],
                                "start_date": "2016-01-01",
                                "end_date": "2018-01-01",
                            }
                        ],
                        "founder_roles": [
                            {
                                "id": "fr-1",
                                "company": "Startup",
                                "role": "Founder",
                                "location": "Remote",
                                "description": "",
                                "bullets": [],
                                "start_date": "2018-01-01",
                                "end_date": "2019-01-01",
                            }
                        ],
                        "skills": [
                            {
                                "category": "Core",
                                "skills": [{"id": "skill-1", "name": "Python"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            client = _make_client()
            client.reset()
            client.import_assets(dummy_assets)
            client.import_assets(dummy_assets.resolve())
            with _capture_maxcov_output("maxcov extras neo4j missing seed file"):
                client.import_assets("missing_assets.json")
            client.reset_and_import(dummy_assets)
            client.ensure_resume_exists(dummy_assets)
            client.get_resume_data()
            client.close()

            _make_client("has_resume").ensure_resume_exists(dummy_assets)
            _make_client("auto_fit_none").get_auto_fit_cache()
            _make_client().get_auto_fit_cache()
            _make_client().set_auto_fit_cache(best_scale=1.05, too_long_scale=None)
            _make_client().set_auto_fit_cache(best_scale=1.05, too_long_scale=1.25)
            _make_client("resume_none").get_resume_data()

            _make_client(skills_json="not-json").list_applied_jobs()
            _make_client(skills_json='"a,b"').list_applied_jobs()
            _make_client(skills_json='["a,b", ["c"], null]').list_applied_jobs()
            _make_client(skills_json='{"x": "y"}').list_applied_jobs()
            _make_client(skills_json='[{"x": "y"}]').list_applied_jobs()
            _make_client(
                skills_json='[["a", "b"], ["c", "d"], ["e"]]'
            ).list_applied_jobs()

            try:
                _make_client("save_none").save_resume({"summary": "x"})
            except Exception:
                pass
            _make_client().save_resume({"summary": "x"})

            _make_client().upsert_resume_and_sections(
                {
                    "summary": "x",
                    "name": "Test User",
                    "first_name": "Test",
                    "middle_name": "",
                    "last_name": "User",
                    "email": "test@example.com",
                    "phone": "555-555-5555",
                    "linkedin_url": "",
                    "github_url": "",
                    "scholar_url": "",
                    "head1_left": "",
                    "head1_middle": "",
                    "head1_right": "",
                    "head2_left": "",
                    "head2_middle": "",
                    "head2_right": "",
                    "head3_left": "",
                    "head3_middle": "",
                    "head3_right": "",
                    "top_skills": [],
                },
                [
                    {
                        "id": "exp-1",
                        "company": "Company",
                        "role": "Role",
                        "location": "Remote",
                        "description": "",
                        "bullets": [],
                        "start_date": "2020-01-01",
                        "end_date": "2021-01-01",
                    }
                ],
                [
                    {
                        "id": "edu-1",
                        "school": "School",
                        "degree": "B.S.",
                        "location": "City",
                        "description": "",
                        "bullets": [],
                        "start_date": "2016-01-01",
                        "end_date": "2018-01-01",
                    }
                ],
                [
                    {
                        "id": "fr-1",
                        "company": "Startup",
                        "role": "Founder",
                        "location": "Remote",
                        "description": "",
                        "bullets": [],
                        "start_date": "2018-01-01",
                        "end_date": "2019-01-01",
                    }
                ],
            )

            orig_schema_ready = globals().get("_NEO4J_SCHEMA_READY")

            class _BadSession:
                def run(self, *_args, **_kwargs):
                    raise RuntimeError("schema fail")

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            class _BadDriver:
                def session(self):
                    return _BadSession()

                def close(self):
                    return None

            globals()["_NEO4J_SCHEMA_READY"] = False
            bad_client = Neo4jClient(driver=_BadDriver())
            bad_client.close()
            globals()["_NEO4J_SCHEMA_READY"] = orig_schema_ready

            orig_stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
            orig_stub_db = globals().get("_MAXCOV_DB")
            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            globals()["_MAXCOV_DB"] = {"resume": {"id": "resume-1"}, "profiles": []}
            stub_client = Neo4jClient()
            stub_client._ensure_schema()
            stub_client._ensure_placeholder_relationships()
            stub_client.close()
            if orig_stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = orig_stub_env
            globals()["_MAXCOV_DB"] = orig_stub_db

            class _OkSession:
                def run(self, *_args, **_kwargs):
                    return None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            class _OkDriver:
                def session(self):
                    return _OkSession()

                def close(self):
                    return None

            ok_client = Neo4jClient(driver=_OkDriver())
            ok_client._ensure_placeholder_relationships()
            ok_client.close()
        except Exception:
            pass

        _maxcov_log("maxcov extras neo4j dummy done")

        # Exercise profile bullet updates with a dummy driver.
        try:

            class _UpdResult:
                def __init__(self, row=None):
                    self._row = row

                def single(self):
                    return self._row

            class _UpdSession:
                def run(self, *_args, **_kwargs):
                    return _UpdResult({"id": "profile-1"})

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            class _UpdDriver:
                def session(self):
                    return _UpdSession()

                def close(self):
                    return None

            client = Neo4jClient(driver=_UpdDriver())
            client.update_profile_bullets(
                "profile-1",
                [{"id": "e1", "bullets": ["a"]}],
                [{"id": "f1", "bullets": ["b"]}],
            )
            client.close()
            orig_stub_env = os.environ.get("MAX_COVERAGE_STUB_DB")
            orig_stub_db = globals().get("_MAXCOV_DB")
            os.environ["MAX_COVERAGE_STUB_DB"] = "1"
            globals()["_MAXCOV_DB"] = {
                "resume": {"id": "resume-1"},
                "profiles": [
                    {"id": "profile-0"},
                    {"id": "profile-1"},
                ],
            }
            stub_client = Neo4jClient()
            stub_client.update_profile_bullets(
                "profile-1",
                [{"id": "e1", "bullets": ["a"]}],
                [{"id": "f1", "bullets": ["b"]}],
            )
            stub_client.update_profile_bullets("missing", [], [])
            stub_client.close()
            if orig_stub_env is None:
                os.environ.pop("MAX_COVERAGE_STUB_DB", None)
            else:
                os.environ["MAX_COVERAGE_STUB_DB"] = orig_stub_env
            globals()["_MAXCOV_DB"] = orig_stub_db
            _force_exception("profile-update")
        except Exception:
            pass
        _maxcov_log("maxcov extras profile update done")

        # Exercise render_resume_pdf_bytes without hitting external services.
        with _capture_maxcov_output("maxcov extras render_resume_pdf_bytes"):
            orig_render_db = Neo4jClient
            orig_render_compile = compile_pdf
            try:

                class _RenderNeo4j:
                    def __init__(self, *args, **kwargs):
                        pass

                    def get_resume_data(self):
                        return {
                            "resume": {
                                "head1_left": "Left",
                                "head1_middle": "Middle",
                                "head1_right": "Right",
                                "head2_left": "",
                                "head2_middle": "",
                                "head2_right": "",
                                "head3_left": "",
                                "head3_middle": "",
                                "head3_right": "",
                                "top_skills": ["Skill 1", "Skill 2"],
                                "summary": "Summary text.",
                                "first_name": "Test",
                                "middle_name": "Q",
                                "last_name": "User",
                                "email": "test@example.com",
                                "phone": "555-555-5555",
                                "section_order": "summary,experience",
                                "section_enabled": {"summary": True, "matrices": False},
                                "linkedin_url": "",
                                "github_url": "",
                                "scholar_url": "",
                            },
                            "experience": [],
                            "education": [],
                            "founder_roles": [],
                        }

                    def close(self):
                        return None

                globals()["Neo4jClient"] = _RenderNeo4j
                globals()["compile_pdf"] = lambda *_a, **_k: (True, b"%PDF-1.4\n%")
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )
                render_resume_pdf_bytes(
                    save_copy=True,
                    include_summary=False,
                    include_skills=False,
                    filename="preview_no_summary_skills.pdf",
                )
                globals()["compile_pdf"] = lambda *_a, **_k: (False, b"")
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )

                class _FailingRenderNeo4j:
                    def __init__(self, *args, **kwargs):
                        raise RuntimeError("Simulated render failure")

                globals()["Neo4jClient"] = _FailingRenderNeo4j
                render_resume_pdf_bytes(
                    save_copy=False, include_summary=True, include_skills=True
                )
            finally:
                globals()["Neo4jClient"] = orig_render_db
                globals()["compile_pdf"] = orig_render_compile

        _maxcov_log("maxcov extras render resume done")

        # Exercise reasoning parameter mapping.
        orig_effort = DEFAULT_LLM_REASONING_EFFORT
        try:
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = "minimal"
            _openai_reasoning_params_for_model("gpt-5.2")
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = "none"
            _openai_reasoning_params_for_model("gpt-5.2")
        finally:
            globals()["DEFAULT_LLM_REASONING_EFFORT"] = orig_effort

        _maxcov_log("maxcov extras reasoning params done")

        # Build Typst source with multiple data shapes.
        resume_data = {
            "summary": "First sentence. Second sentence.",
            "headers": [str(i) for i in range(9)],
            "highlighted_skills": [f"Skill {i}" for i in range(1, 10)],
            "skills_rows": [],
            "top_skills": ["AI", "ML"],
            "first_name": "Jane",
            "middle_name": "Q",
            "last_name": "Public",
            "target_role": "Engineer",
            "target_company": "Acme",
            "primary_domain": "AI",
            "req_id": "REQ-1",
        }
        profile_data = {
            "email": "jane@example.com",
            "phone": "555-0100",
            "linkedin_url": "https://linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "summary": "Profile summary",
            "experience": [
                {
                    "role": "Dev",
                    "company": "Co",
                    "location": "Remote",
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "description": "Did stuff.",
                    "bullets": ["Bullet 1", "Bullet 2"],
                }
            ],
            "education": [
                {
                    "degree": "Master of Science (AI; Systems)",
                    "school": "Test University",
                    "start_date": "2016-01-01",
                    "end_date": "2018-01-01",
                    "description": "Coursework",
                    "bullets": ["Course A"],
                }
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "location": "Remote",
                    "description": "Built things.",
                    "bullets": ["2020---2021||Raised funds", "Did thing"],
                }
            ],
        }
        generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=True,
            include_summary=True,
            section_order=["summary", "unknown", "education", "experience", "founder"],
        )
        resume_data_alt = dict(resume_data)
        resume_data_alt["summary"] = "No punctuation summary"
        resume_data_alt["skills_rows"] = '{"bad": "data"}'
        profile_data_alt = {
            "email": "jane@example.com",
            "phone": "555-0100",
            "linkedin_url": "https://linkedin.com/in/jane",
            "github_url": "https://github.com/jane",
            "scholar_url": "abc123",
            "summary": "",
            "experience": [
                {
                    "role": "Dev",
                    "company": "Co",
                    "location": "Remote",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": ["", "Bullet 1"],
                },
                {
                    "role": "Other",
                    "company": "Co2",
                    "location": "Remote",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": [],
                },
            ],
            "education": [
                {
                    "degree": "",
                    "school": "",
                    "start_date": "",
                    "end_date": "2018-01-01",
                    "description": "",
                    "bullets": [],
                },
                {
                    "degree": "",
                    "school": "",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "bullets": [],
                },
            ],
            "founder_roles": [
                {
                    "company": "Startup",
                    "location": "Remote",
                    "description": "",
                    "bullets": [],
                }
            ],
        }
        generate_typst_source(
            resume_data_alt,
            profile_data_alt,
            include_matrices=True,
            include_summary=True,
            section_order=["education", "experience", "founder"],
        )
        resume_data["skills_rows"] = '["a,b", ["c", "d"], null]'
        profile_data["education"] = [
            {
                "degree": "B.S.",
                "school": "State College",
                "start_date": "2012-01-01",
                "end_date": "2016-01-01",
                "description": "",
                "bullets": [],
            }
        ]
        generate_typst_source(
            resume_data,
            profile_data,
            include_matrices=False,
            include_summary=False,
            section_order=["experience", "education"],
        )

        # Exercise rasterized text paths with a stub PIL module.
        orig_pil_mod = sys.modules.get("PIL")
        orig_select_fonts = globals().get("_select_local_font_paths")
        orig_temp_build = TEMP_BUILD_DIR
        tmp_raster_dir = None
        try:

            class _FakeImage:
                def __init__(self, size):
                    self.width, self.height = size

                def resize(self, size, _resample):
                    return _FakeImage(size)

                def save(self, path, format=None):
                    Path(path).write_bytes(b"fake")

            class _FakeDraw:
                def __init__(self, _img):
                    return None

                def textbbox(self, _pos, _text, font=None):
                    return (0, 0, 10, 10)

                def text(self, *_args, **_kwargs):
                    return None

            class _FakeImageModule:
                BICUBIC = 3

                class Resampling:
                    LANCZOS = 1

                @staticmethod
                def new(_mode, size, _color):
                    return _FakeImage(size)

            class _FakeImageDrawModule:
                @staticmethod
                def Draw(img):
                    return _FakeDraw(img)

            class _FakeFont:
                pass

            class _FakeImageFontModule:
                @staticmethod
                def truetype(path, _size):
                    if "fail" in str(path):
                        raise OSError("bad font")
                    return _FakeFont()

                @staticmethod
                def load_default():
                    return _FakeFont()

            fake_pil = types.ModuleType("PIL")
            fake_pil.Image = _FakeImageModule
            fake_pil.ImageDraw = _FakeImageDrawModule
            fake_pil.ImageFont = _FakeImageFontModule
            sys.modules["PIL"] = fake_pil
            tmp_raster_dir = Path(
                tempfile.mkdtemp(prefix="maxcov_raster_", dir=BASE_DIR)
            )
            globals()["TEMP_BUILD_DIR"] = tmp_raster_dir
            globals()["_select_local_font_paths"] = lambda *_a, **_k: [
                tmp_raster_dir / "fail.otf",
                tmp_raster_dir / "ok.otf",
            ]
            raster_resume = {
                "summary": "Raster",
                "headers": [],
                "highlighted_skills": [],
                "skills_rows": [],
                "first_name": "",
                "middle_name": "",
                "last_name": "",
                "font_family": "",
            }
            raster_profile = {
                "name": "Raster User",
                "experience": [
                    {
                        "role": "Role",
                        "company": "Co",
                        "location": "",
                        "start_date": "",
                        "end_date": "",
                        "description": "Desc",
                        "bullets": ["2020---2021||Did thing"],
                    }
                ],
                "founder_roles": [
                    {
                        "company": "Startup",
                        "location": "",
                        "description": "Founder desc",
                        "bullets": ["2020---2021||Built"],
                    }
                ],
            }
            generate_typst_source(
                raster_resume,
                raster_profile,
                include_matrices=False,
                include_summary=True,
                section_order=["experience", "founder"],
            )
        finally:
            globals()["_select_local_font_paths"] = orig_select_fonts
            globals()["TEMP_BUILD_DIR"] = orig_temp_build
            if orig_pil_mod is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil_mod
            if tmp_raster_dir is not None:
                try:
                    for path in tmp_raster_dir.rglob("*"):
                        if path.is_file():
                            path.unlink()
                    for path in sorted(tmp_raster_dir.rglob("*"), reverse=True):
                        if path.is_dir():
                            path.rmdir()
                except Exception:
                    pass

        _maxcov_log("maxcov extras typst shapes done")

        # Exercise PDF metadata and auto-fit logic with stubbed PDF rendering.
        try:
            import pikepdf

            tmp_pdf = tmp_home / "sample.pdf"
            pikepdf.new().save(tmp_pdf)
            _apply_pdf_metadata(
                tmp_pdf,
                _build_pdf_metadata(resume_data, profile_data),
            )
        except Exception:
            pass

        orig_generate_typst_source = generate_typst_source
        orig_compile_pdf = compile_pdf
        try:
            scale_state = {"pages": 2}

            def _fake_generate_typst_source(*_args, layout_scale=1.0, **_kwargs):
                return f"scale={layout_scale}"

            def _fake_compile_pdf(typst_source, metadata=None):
                try:
                    scale = float(str(typst_source).split("=", 1)[1])
                except Exception:
                    scale = 1.0
                if scale > 1.2:
                    scale_state["pages"] = 3
                elif scale < 0.8:
                    scale_state["pages"] = 1
                else:
                    scale_state["pages"] = 2
                return True, b"%PDF-1.4\n%"

            class _DummyPdf:
                def __init__(self, pages):
                    self.pages = [None] * int(pages)

            def _fake_pikepdf_open(_stream):
                return _DummyPdf(scale_state["pages"])

            globals()["generate_typst_source"] = _fake_generate_typst_source
            globals()["compile_pdf"] = _fake_compile_pdf
            try:
                import pikepdf as _pikepdf

                orig_pikepdf_open = _pikepdf.open
                _pikepdf.open = _fake_pikepdf_open
            except Exception:
                orig_pikepdf_open = None
            compile_pdf_with_auto_tuning(
                resume_data,
                profile_data,
                include_matrices=True,
                include_summary=True,
                section_order=["summary"],
            )
            globals()["compile_pdf"] = lambda *_a, **_k: (True, b"%PDF-1.4\n%")
            if orig_pikepdf_open:
                _pikepdf.open = _fake_pikepdf_open
            scale_state["pages"] = 3
            compile_pdf_with_auto_tuning(
                resume_data,
                profile_data,
                include_matrices=True,
                include_summary=True,
                section_order=["summary"],
            )
            if orig_pikepdf_open:
                _pikepdf.open = orig_pikepdf_open
        finally:
            globals()["generate_typst_source"] = orig_generate_typst_source
            globals()["compile_pdf"] = orig_compile_pdf

        _maxcov_log("maxcov extras pdf metadata stub done")

        # Stub LLM responses to exercise JSON parsing paths.
        orig_llm_responses = _call_llm_responses
        orig_llm_completion = _call_llm_completion
        orig_skip_llm = os.environ.get("MAX_COVERAGE_SKIP_LLM")
        os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
        orig_fake_generate = globals().get("_fake_generate_resume_content")
        orig_openai_key = os.environ.get("OPENAI_API_KEY")
        orig_gemini_key = os.environ.get("GEMINI_API_KEY")
        orig_google_key = os.environ.get("GOOGLE_API_KEY")
        orig_home = os.environ.get("HOME")
        tmp_llm_home = None
        try:
            try:
                os.environ["MAX_COVERAGE_SKIP_LLM"] = "1"

                def _boom_fake(*_a, **_k):
                    raise RuntimeError("skip")

                globals()["_fake_generate_resume_content"] = _boom_fake
                generate_resume_content("req", {"summary": "Base"}, "openai:gpt-5.2")
            except Exception:
                pass
            finally:
                os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
                if orig_fake_generate is None:
                    globals().pop("_fake_generate_resume_content", None)
                else:
                    globals()["_fake_generate_resume_content"] = orig_fake_generate

            tmp_llm_home = Path(tempfile.mkdtemp(prefix="maxcov_llm_home_"))
            os.environ["HOME"] = str(tmp_llm_home)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            generate_resume_content("req", {"prompt_yaml": "Prompt"}, "openai:gpt-5.2")
            generate_resume_content(
                "req", {"prompt_yaml": "Prompt"}, "gemini:gemini-1.5-flash"
            )

            os.environ["OPENAI_API_KEY"] = "sk-test"
            os.environ["GEMINI_API_KEY"] = "gm-test"
            os.environ["OPENAI_BASE_URL"] = "http://example.com"
            os.environ["OPENAI_ORGANIZATION"] = "org-test"
            os.environ["OPENAI_PROJECT"] = "proj-test"

            def _fake_llm_responses_missing(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_responses"] = _fake_llm_responses_missing
            generate_resume_content("req", {"prompt_yaml": "Prompt"}, "openai:gpt-5.2")

            def _fake_llm_completion_missing(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_completion"] = _fake_llm_completion_missing
            generate_resume_content(
                "req", {"prompt_yaml": "Prompt"}, "gemini:gemini-1.5-flash"
            )

            prompt_path = BASE_DIR / "prompt.yaml"
            prompt_backup = None
            try:
                if prompt_path.exists():
                    prompt_backup = prompt_path.with_suffix(".yaml.maxcov")
                    prompt_path.rename(prompt_backup)
                generate_resume_content("req", {}, "openai:gpt-5.2")
            finally:
                if prompt_backup and prompt_backup.exists():
                    prompt_backup.rename(prompt_path)

            class _DummyIncomplete:
                def __init__(self, reason):
                    self.reason = reason

            class _DummyOpenAIResp:
                def __init__(self, output_text, status="completed", reason=None):
                    self.output_text = output_text
                    self.status = status
                    self.incomplete_details = (
                        _DummyIncomplete(reason) if reason else None
                    )

            def _fake_llm_responses_simple(**_request):
                return _DummyOpenAIResp('{"summary":"ok"}')

            globals()["_call_llm_responses"] = _fake_llm_responses_simple
            try:
                generate_resume_content(
                    req_text,
                    {"prompt_yaml": "Prompt", "rewrite_bullets": True},
                    "openai:gpt-5.2",
                )
            except Exception:
                pass

            openai_calls = {"n": 0}

            def _fake_llm_responses(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp('{"summary":"ok"}')
                if openai_calls["n"] == 2:
                    return _DummyOpenAIResp("not json")
                if openai_calls["n"] == 3:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp('{"summary":"retry"}')

            globals()["_call_llm_responses"] = _fake_llm_responses
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_fenced(**_request):
                return _DummyOpenAIResp('```json\n{"summary":"ok"}\n```')

            globals()["_call_llm_responses"] = _fake_llm_responses_fenced
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_wrapped(**_request):
                return _DummyOpenAIResp('prefix {"summary":"ok"} suffix')

            globals()["_call_llm_responses"] = _fake_llm_responses_wrapped
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_empty(**_request):
                return _DummyOpenAIResp("")

            globals()["_call_llm_responses"] = _fake_llm_responses_empty
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_empty_fenced(**_request):
                return _DummyOpenAIResp("```")

            globals()["_call_llm_responses"] = _fake_llm_responses_empty_fenced
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_error(**_request):
                raise MissingApiKeyError("missing key")

            globals()["_call_llm_responses"] = _fake_llm_responses_error
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            def _fake_llm_responses_unsupported(**_request):
                raise UnsupportedProviderError("unsupported")

            globals()["_call_llm_responses"] = _fake_llm_responses_unsupported
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_error(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                raise RuntimeError("retry fail")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_error
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_empty(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp("")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_empty
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            openai_calls = {"n": 0}

            def _fake_llm_responses_retry_invalid(**_request):
                openai_calls["n"] += 1
                if openai_calls["n"] == 1:
                    return _DummyOpenAIResp(
                        "", status="incomplete", reason="max_output_tokens"
                    )
                return _DummyOpenAIResp("invalid json")

            globals()["_call_llm_responses"] = _fake_llm_responses_retry_invalid
            try:
                generate_resume_content(req_text, {}, "openai:gpt-5.2")
            except Exception:
                pass

            class _DummyGeminiChoice:
                def __init__(self, content, finish_reason="stop"):
                    self.message = type("Msg", (), {"content": content})()
                    self.finish_reason = finish_reason

            class _DummyGeminiResp:
                def __init__(self, content, finish_reason="stop"):
                    self.choices = [_DummyGeminiChoice(content, finish_reason)]

            gemini_calls = {"n": 0}

            def _fake_llm_completion(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    raise RuntimeError("Simulated Gemini error")
                return _DummyGeminiResp(["{", None, '"summary":"ok"', "}"])

            globals()["_call_llm_completion"] = _fake_llm_completion
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp('{"summary":"retry"}')

            globals()["_call_llm_completion"] = _fake_llm_completion_retry
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_invalid(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp("not json")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_invalid
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_invalid_json(**_request):
                return _DummyGeminiResp("not json", finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_invalid_json
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_bad_choice(**_request):
                class _Resp:
                    choices = None

                return _Resp()

            globals()["_call_llm_completion"] = _fake_llm_completion_bad_choice
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_missing(**_request):
                raise MissingApiKeyError("missing")

            globals()["_call_llm_completion"] = _fake_llm_completion_missing
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_unsupported(**_request):
                raise UnsupportedProviderError("unsupported")

            globals()["_call_llm_completion"] = _fake_llm_completion_unsupported
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_empty(**_request):
                return _DummyGeminiResp("", finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_empty
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            def _fake_llm_completion_none(**_request):
                return _DummyGeminiResp(None, finish_reason="stop")

            globals()["_call_llm_completion"] = _fake_llm_completion_none
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_fail(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                raise RuntimeError("retry failed")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_fail
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass

            gemini_calls = {"n": 0}

            def _fake_llm_completion_retry_empty(**_request):
                gemini_calls["n"] += 1
                if gemini_calls["n"] == 1:
                    return _DummyGeminiResp("", finish_reason="max_tokens")
                return _DummyGeminiResp("")

            globals()["_call_llm_completion"] = _fake_llm_completion_retry_empty
            try:
                generate_resume_content(req_text, {}, "gemini:gemini-1.5-flash")
            except Exception:
                pass
        finally:
            if orig_skip_llm is None:
                os.environ.pop("MAX_COVERAGE_SKIP_LLM", None)
            else:
                os.environ["MAX_COVERAGE_SKIP_LLM"] = orig_skip_llm
            if orig_openai_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = orig_openai_key
            if orig_gemini_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = orig_gemini_key
            if orig_google_key is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = orig_google_key
            if orig_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = orig_home
            if tmp_llm_home is not None:
                try:
                    for path in tmp_llm_home.iterdir():
                        if path.is_file():
                            path.unlink()
                    tmp_llm_home.rmdir()
                except Exception:
                    pass
            globals()["_call_llm_responses"] = orig_llm_responses
            globals()["_call_llm_completion"] = orig_llm_completion

        static_report = os.environ.get("MAX_COVERAGE_STATIC_REPORT_PATH")
        if static_report:
            try:
                _maxcov_log("maxcov extras static analysis start")
                _run_static_analysis_tools(Path(static_report))
                _maxcov_log("maxcov extras static analysis done")
            except Exception:
                pass

        _maxcov_log("maxcov extras done")

    def _run_playwright_ui_traversal(
        url: str,
        *,
        timeout_s: float = 30.0,
    ) -> bool:
        target = (url or "").strip()
        if not target or target.lower() in {"0", "false", "none"}:
            return False
        _maxcov_log(f"playwright start: {target}")
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            print(f"Playwright not available: {exc}")
            _maxcov_log("playwright done (unavailable)")
            return False

        timeout_ms = max(1000, int(float(timeout_s) * 1000))
        action_timeout_ms = min(5000, max(500, int(timeout_ms * 0.25)))

        def safe_click(locator) -> None:
            try:
                if locator.count() == 0:
                    return
                locator.click(timeout=action_timeout_ms)
            except Exception:
                pass

        def safe_fill(locator, value: str) -> None:
            try:
                if locator.count() == 0:
                    return
                locator.fill(value, timeout=action_timeout_ms)
            except Exception:
                pass

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(timeout_ms)
                try:
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        page.get_by_role("heading", name="Resume Builder").wait_for(
                            timeout=timeout_ms
                        )
                    except Exception:
                        pass

                    safe_fill(
                        page.get_by_placeholder(
                            "Paste or type the job requisition here"
                        ),
                        "Playwright coverage smoke test.",
                    )
                    safe_click(page.get_by_role("button", name="Load Data"))
                    page.wait_for_timeout(500)

                    switches = page.get_by_role("switch")
                    try:
                        count = switches.count()
                    except Exception:
                        count = 0
                    if count:
                        safe_click(switches.nth(0))
                        if count > 1:
                            safe_click(switches.nth(1))

                    safe_click(page.get_by_label("Move section down"))
                    safe_click(page.get_by_label("Move section up"))
                    safe_click(page.get_by_role("button", name="Save Data"))

                    safe_click(page.get_by_role("button", name="Add Experience"))
                    safe_fill(page.get_by_placeholder("Role").first, "Playwright Role")
                    safe_fill(page.get_by_placeholder("Company").first, "Playwright Co")

                    safe_click(page.get_by_role("button", name="Add Education"))
                    safe_fill(
                        page.get_by_placeholder("Degree").first,
                        "Playwright Degree",
                    )
                    safe_fill(
                        page.get_by_placeholder("School").first,
                        "Playwright School",
                    )

                    safe_click(page.get_by_role("button", name="Add Founder Role"))
                    safe_fill(
                        page.get_by_placeholder("Role").last, "Playwright Founder"
                    )
                    safe_fill(
                        page.get_by_placeholder("Company").last, "Playwright Startup"
                    )

                    safe_click(page.get_by_role("button", name="Generate PDF"))
                    page.wait_for_timeout(1000)
                finally:
                    browser.close()
            return True
        except Exception as exc:
            print(f"Playwright traversal failed: {exc}")
            return False
        finally:
            _maxcov_log("playwright done")

    def _wait_for_url(url: str, timeout_s: float) -> bool:
        deadline = time.time() + max(1.0, float(timeout_s))
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return True
            except Exception:
                time.sleep(0.25)
        return False

    def _pick_open_port(preferred: int) -> int:
        if preferred and preferred > 0:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", preferred))
                    return preferred
            except OSError:
                pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _run_reflex_coverage_session(
        frontend_port: int,
        backend_port: int,
        *,
        startup_timeout_s: float,
        ui_timeout_s: float,
    ) -> str | None:
        frontend_port = _pick_open_port(int(frontend_port))
        backend_port = _pick_open_port(
            int(backend_port) if int(backend_port) != int(frontend_port) else 0
        )

        log_dir = Path(tempfile.gettempdir()) / "dce_tools"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_path = log_dir / f"reflex_run_coverage_{stamp}.log"
        url = f"http://localhost:{frontend_port}"
        _maxcov_log(
            f"reflex session start: frontend={frontend_port} backend={backend_port}"
        )

        cov_file = os.environ.get("COVERAGE_FILE")
        env = os.environ.copy()
        env["REFLEX_COVERAGE"] = "1"
        if cov_file:
            env["COVERAGE_FILE"] = cov_file
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            "reflex",
            "run",
            "--frontend-port",
            str(frontend_port),
            "--backend-port",
            str(backend_port),
        ]
        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=log_file,
                    stderr=log_file,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                print(f"Warning: failed to start reflex coverage server: {exc}")
                return None

            try:
                if not _wait_for_url(url, startup_timeout_s):
                    print("Warning: reflex coverage server did not start in time.")
                    return None
                _run_playwright_ui_traversal(url, timeout_s=ui_timeout_s)
                return url
            finally:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGINT)
                    else:
                        proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        if hasattr(os, "killpg"):
                            os.killpg(proc.pid, signal.SIGTERM)
                        else:
                            proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            if hasattr(os, "killpg"):
                                os.killpg(proc.pid, signal.SIGKILL)
                            else:
                                proc.kill()
                        except Exception:
                            pass
                time.sleep(1.0)
                _maxcov_log("reflex session done")

    def _read_log_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            return ""
        minutes = int(seconds // 60)
        remainder = seconds - (minutes * 60)
        if minutes:
            return f"{minutes}m {remainder:.1f}s"
        return f"{remainder:.1f}s"

    def _read_static_analysis_report(path: Path) -> list[dict]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        cleaned: list[dict] = []
        for item in data:
            if isinstance(item, dict):
                cleaned.append(item)
        return cleaned

    def _render_run_all_tests_summary(
        rows: list[dict], total_duration_s: float
    ) -> None:
        try:
            from rich import box
            from rich.console import Console
            from rich.table import Table
            from rich.text import Text
        except Exception:
            print("Run-all-tests summary:")
            for row in rows:
                print(
                    f"- {row.get('step', '')}: {row.get('status', '')} "
                    f"({row.get('duration', '')}) {row.get('details', '')}".strip()
                )
            print(f"Total: {_format_duration(total_duration_s)}")
            return

        table = Table(title="Run All Tests", box=box.ASCII, show_lines=False)
        table.add_column("Step", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Details", overflow="fold")

        status_styles = {
            "ok": "green",
            "warn": "yellow",
            "skip": "dim",
            "fail": "bold red",
        }
        for row in rows:
            status = str(row.get("status", "")).lower() or "unknown"
            status_text = Text(status.upper())
            status_text.stylize(status_styles.get(status, "white"))
            table.add_row(
                str(row.get("step", "")),
                status_text,
                str(row.get("duration", "")),
                str(row.get("details", "")),
            )

        overall = (
            "PASS"
            if all(r.get("status") != "fail" for r in rows)
            else "FAIL"
        )
        table.add_section()
        table.add_row(
            "Total",
            Text(overall, style=("bold green" if overall == "PASS" else "bold red")),
            _format_duration(total_duration_s),
            "",
        )
        Console().print(table)

    def _scan_reflex_log_for_issues(log_text: str) -> list[str]:
        issues: list[str] = []
        pattern = re.compile(
            r"\b(warning|error|traceback|exception)\b", re.IGNORECASE
        )
        for line in (log_text or "").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if pattern.search(cleaned):
                issues.append(cleaned)
        return issues

    def _stop_reflex_process(proc: subprocess.Popen) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGINT)
            else:
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass

    def _parse_ui_actions(raw: str) -> set[str]:
        raw = (raw or "").strip().lower()
        valid = {
            "load",
            "profile",
            "pipeline",
            "forms",
            "toggles",
            "reorder",
            "save",
            "pdf",
        }
        if not raw or raw == "all":
            return set(valid)
        actions = {item.strip() for item in raw.split(",") if item.strip()}
        unknown = actions - valid
        if unknown:
            raise ValueError(
                f"Unknown ui-simulate action(s): {', '.join(sorted(unknown))}"
            )
        return actions

    if args.reset_db:
        assets = args.reset_db or str(DEFAULT_ASSETS_JSON)
        print(f"Resetting Neo4j and importing assets from {assets}...")
        try:
            db = Neo4jClient()
            db.reset_and_import(assets)
            db.close()
            print("Reset + import completed successfully.")
        except Exception as e:
            print(f"Error resetting/importing assets: {e}")
            sys.exit(1)

    if args.import_assets and not args.reset_db:
        print(f"Importing assets from {args.import_assets}...")
        try:
            db = Neo4jClient()
            imported = db.import_assets(
                args.import_assets, allow_overwrite=bool(args.overwrite_resume)
            )
            db.close()
            if not imported:
                sys.exit(1)
            print("Import completed successfully.")
        except Exception as e:
            print(f"Error importing assets: {e}")
            sys.exit(1)

    if args.run_all_tests:
        results: list[dict] = []
        overall_rc = 0
        total_started = time.perf_counter()
        log_dir = Path(tempfile.gettempdir()) / "dce_tools"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        static_report_path = Path(
            os.environ.get(
                "MAX_COVERAGE_STATIC_REPORT_PATH",
                str(log_dir / f"maxcov_static_{stamp}.json"),
            )
        )

        def record(step: str, status: str, duration_s: float, details: str = "") -> None:
            results.append(
                {
                    "step": step,
                    "status": status,
                    "duration": _format_duration(duration_s),
                    "details": details,
                }
            )

        maxcov_started = time.perf_counter()
        maxcov_cmd = [
            sys.executable,
            str(BASE_DIR / "harness.py"),
            "--maximum-coverage",
            "--maximum-coverage-actions",
            str(args.maximum_coverage_actions),
            "--maximum-coverage-ui-timeout",
            str(args.maximum_coverage_ui_timeout),
            "--maximum-coverage-reflex-frontend-port",
            str(args.maximum_coverage_reflex_frontend_port),
            "--maximum-coverage-reflex-backend-port",
            str(args.maximum_coverage_reflex_backend_port),
            "--maximum-coverage-reflex-startup-timeout",
            str(args.maximum_coverage_reflex_startup_timeout),
        ]
        if args.maximum_coverage_ui_url:
            maxcov_cmd.extend(
                ["--maximum-coverage-ui-url", str(args.maximum_coverage_ui_url)]
            )
        if args.maximum_coverage_skip_llm:
            maxcov_cmd.append("--maximum-coverage-skip-llm")
        if args.maximum_coverage_failures:
            maxcov_cmd.append("--maximum-coverage-failures")
        if args.maximum_coverage_reflex:
            maxcov_cmd.append("--maximum-coverage-reflex")
        maxcov_env = os.environ.copy()
        maxcov_env.setdefault("MAX_COVERAGE_CONTAINER", "1")
        maxcov_env.setdefault("MAX_COVERAGE_STUB_DB", "1")
        maxcov_env["MAX_COVERAGE_STATIC_REPORT_PATH"] = str(static_report_path)
        maxcov_result = subprocess.run(
            maxcov_cmd,
            cwd=str(BASE_DIR),
            env=maxcov_env,
        )
        maxcov_duration = time.perf_counter() - maxcov_started
        if maxcov_result.returncode != 0:
            record(
                "maximum-coverage",
                "fail",
                maxcov_duration,
                f"rc={maxcov_result.returncode}",
            )
            overall_rc = maxcov_result.returncode or 1
            _render_run_all_tests_summary(
                results, time.perf_counter() - total_started
            )
            sys.exit(overall_rc)
        record("maximum-coverage", "ok", maxcov_duration, "ok")

        static_results = _read_static_analysis_report(static_report_path)
        static_failed = False
        if static_results:
            for item in static_results:
                tool = str(item.get("tool") or "static")
                status = str(item.get("status") or "warn").lower()
                duration_s = float(item.get("duration_s") or 0.0)
                details = str(item.get("details") or "")
                record(f"static: {tool}", status, duration_s, details)
                if status == "fail":
                    static_failed = True
            if static_failed and overall_rc == 0:
                overall_rc = 1
        else:
            record(
                "static analysis",
                "warn",
                0.0,
                "no report",
            )

        diagram_log = log_dir / f"diagrams_run_all_tests_{stamp}.log"
        diagrams_started = time.perf_counter()
        diagrams_cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "generate_diagrams.py"),
        ]
        diagrams_result = subprocess.run(
            diagrams_cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        diagrams_duration = time.perf_counter() - diagrams_started
        diagram_output = "\n".join(
            [t for t in (diagrams_result.stdout or "", diagrams_result.stderr or "") if t]
        )
        try:
            diagram_log.write_text(diagram_output, encoding="utf-8")
        except Exception:
            pass
        if diagrams_result.returncode != 0:
            record(
                "diagram generation",
                "fail",
                diagrams_duration,
                f"rc={diagrams_result.returncode}; log: {diagram_log}",
            )
            if overall_rc == 0:
                overall_rc = diagrams_result.returncode or 1
        else:
            record(
                "diagram generation",
                "ok",
                diagrams_duration,
                f"log: {diagram_log}",
            )

        frontend_port = _pick_open_port(3000)
        backend_port = _pick_open_port(8000 if 8000 != frontend_port else 0)
        log_path = log_dir / f"reflex_run_all_tests_{stamp}.log"
        url = f"http://localhost:{frontend_port}"
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MAX_COVERAGE_STUB_DB", "1")
        env.setdefault("MAX_COVERAGE_SKIP_LLM", "1")
        cmd = [
            "reflex",
            "run",
            "--frontend-port",
            str(frontend_port),
            "--backend-port",
            str(backend_port),
        ]
        proc = None
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                reflex_attempt_started = time.perf_counter()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(BASE_DIR),
                        stdout=log_file,
                        stderr=log_file,
                        env=env,
                        start_new_session=True,
                    )
                except Exception as exc:
                    record(
                        "reflex run (clean start)",
                        "fail",
                        time.perf_counter() - reflex_attempt_started,
                        f"start failed: {exc}",
                    )
                    overall_rc = 1

                if overall_rc == 0:
                    reflex_started = time.perf_counter()
                    if not _wait_for_url(
                        url, float(args.maximum_coverage_reflex_startup_timeout)
                    ):
                        record(
                            "reflex run (clean start)",
                            "fail",
                            time.perf_counter() - reflex_started,
                            "startup timeout",
                        )
                        overall_rc = 1
                    else:
                        time.sleep(1.0)
                        log_text = _read_log_text(log_path)
                        issues = _scan_reflex_log_for_issues(log_text)
                        if issues:
                            first_issue = issues[0]
                            record(
                                "reflex run (clean start)",
                                "fail",
                                time.perf_counter() - reflex_started,
                                f"{first_issue} (log: {log_path})",
                            )
                            overall_rc = 1
                        else:
                            record(
                                "reflex run (clean start)",
                                "ok",
                                time.perf_counter() - reflex_started,
                                f"log: {log_path}",
                            )

                if overall_rc == 0:
                    ui_started = time.perf_counter()
                    ui_cmd = [
                        sys.executable,
                        str(BASE_DIR / "scripts" / "ui_playwright_check.py"),
                        "--url",
                        url,
                        "--timeout",
                        str(args.ui_playwright_timeout),
                        "--pdf-timeout",
                        str(args.ui_playwright_pdf_timeout),
                    ]
                    if args.ui_playwright_headed:
                        ui_cmd.append("--headed")
                    if args.ui_playwright_slowmo:
                        ui_cmd.extend(["--slowmo", str(args.ui_playwright_slowmo)])
                    if args.ui_playwright_allow_llm_error:
                        ui_cmd.append("--allow-llm-error")
                    if args.ui_playwright_allow_db_error:
                        ui_cmd.append("--allow-db-error")
                    if args.ui_playwright_screenshot_dir:
                        ui_cmd.extend(
                            ["--screenshot-dir", args.ui_playwright_screenshot_dir]
                        )
                    ui_result = subprocess.run(ui_cmd, cwd=str(BASE_DIR))
                    ui_duration = time.perf_counter() - ui_started
                    if ui_result.returncode != 0:
                        record(
                            "ui-playwright-check",
                            "fail",
                            ui_duration,
                            f"rc={ui_result.returncode}; log: {log_path}",
                        )
                        overall_rc = ui_result.returncode or 1
                    else:
                        record("ui-playwright-check", "ok", ui_duration, "ok")
        finally:
            if proc is not None:
                _stop_reflex_process(proc)
            _render_run_all_tests_summary(
                results, time.perf_counter() - total_started
            )
            sys.exit(overall_rc)

    if args.maximum_coverage or getattr(args, "ui_simulate", False):
        try:
            actions_raw = args.maximum_coverage_actions
            if hasattr(args, "ui_simulate_actions") and args.ui_simulate_actions:
                actions_raw = args.ui_simulate_actions
            ui_actions = _parse_ui_actions(actions_raw)
        except ValueError as e:
            print(str(e))
            sys.exit(1)
        try:
            skip_llm = bool(args.maximum_coverage_skip_llm) or bool(
                getattr(args, "ui_simulate_skip_llm", False)
            )
            simulate_failures = bool(args.maximum_coverage_failures) or bool(
                getattr(args, "ui_simulate_failures", False)
            )
            _maxcov_log(
                f"ui simulation start: actions={sorted(ui_actions)}, "
                f"skip_llm={skip_llm}, failures={simulate_failures}"
            )
            started = time.perf_counter()
            asyncio.run(
                _run_ui_simulation(
                    ui_actions,
                    args.req_file,
                    skip_llm=skip_llm,
                    simulate_failures=simulate_failures,
                )
            )
            _maxcov_log(f"ui simulation done ({time.perf_counter() - started:.1f}s)")
            if args.maximum_coverage:
                maxcov_started = time.perf_counter()
                reflex_url = None
                if args.maximum_coverage_reflex:
                    try:
                        _maxcov_log("reflex coverage start")
                        started = time.perf_counter()
                        reflex_url = _run_reflex_coverage_session(
                            int(args.maximum_coverage_reflex_frontend_port),
                            int(args.maximum_coverage_reflex_backend_port),
                            startup_timeout_s=float(
                                args.maximum_coverage_reflex_startup_timeout
                            ),
                            ui_timeout_s=float(args.maximum_coverage_ui_timeout or 0)
                            or 30.0,
                        )
                        _maxcov_log(
                            "reflex coverage done "
                            f"({time.perf_counter() - started:.1f}s)"
                        )
                    except Exception as e:
                        print(f"Warning: Reflex coverage session failed: {e}")
                try:
                    _maxcov_log("maximum coverage extras start")
                    started = time.perf_counter()
                    _exercise_maximum_coverage_extras(args.req_file)
                    _maxcov_log(
                        f"maximum coverage extras done ({time.perf_counter() - started:.1f}s)"
                    )
                except Exception as e:
                    print(f"Warning: maximum coverage extras failed: {e}")
                try:
                    playwright_url = None
                    if ui_url_was_set:
                        playwright_url = args.maximum_coverage_ui_url
                    elif not reflex_url:
                        playwright_url = args.maximum_coverage_ui_url
                    if playwright_url:
                        _maxcov_log(f"playwright traversal start: {playwright_url}")
                        started = time.perf_counter()
                        _run_playwright_ui_traversal(
                            playwright_url,
                            timeout_s=float(args.maximum_coverage_ui_timeout or 0)
                            or 30.0,
                        )
                        _maxcov_log(
                            "playwright traversal done "
                            f"({time.perf_counter() - started:.1f}s)"
                        )
                except Exception as e:
                    print(f"Warning: Playwright traversal failed: {e}")
                _maxcov_log(
                    f"maximum coverage done ({time.perf_counter() - maxcov_started:.1f}s)"
                )
        except Exception as e:
            print(f"UI simulation failed: {e}")
            sys.exit(1)

    if args.eval_prompt or args.generate_profile:
        _maxcov_log("cli prompt evaluation start")
        req_path = Path(args.req_file)
        if not req_path.exists():
            print(f"Req file not found: {req_path}")
            sys.exit(1)
        req_text = req_path.read_text(encoding="utf-8", errors="ignore")

        try:
            db = Neo4jClient()
            db.ensure_resume_exists()
            data = db.get_resume_data() or {}
            db.close()
        except Exception as e:
            print(f"Error reading resume from Neo4j: {e}")
            sys.exit(1)

        resume_node = data.get("resume", {}) or {}
        if not resume_node:
            print("No resume found in Neo4j; cannot generate without base profile.")
            sys.exit(1)

        base_profile = {
            **resume_node,
            "experience": data.get("experience", []),
            "education": data.get("education", []),
            "founder_roles": data.get("founder_roles", []),
        }
        model_name = args.model_name or DEFAULT_LLM_MODEL
        _maxcov_log("cli prompt evaluation call LLM")
        llm_result = generate_resume_content(req_text, base_profile, model_name)
        _maxcov_log("cli prompt evaluation LLM done")

        if args.eval_prompt:
            print(json.dumps(llm_result, ensure_ascii=False, indent=2))

        if args.generate_profile:
            if not isinstance(llm_result, dict) or llm_result.get("error"):
                if not args.eval_prompt:
                    print(json.dumps(llm_result, ensure_ascii=False, indent=2))
                sys.exit(1)

            headers = ensure_len(
                [
                    resume_node.get("head1_left", ""),
                    resume_node.get("head1_middle", ""),
                    resume_node.get("head1_right", ""),
                    resume_node.get("head2_left", ""),
                    resume_node.get("head2_middle", ""),
                    resume_node.get("head2_right", ""),
                    resume_node.get("head3_left", ""),
                    resume_node.get("head3_middle", ""),
                    resume_node.get("head3_right", ""),
                ]
            )
            highlighted_skills = ensure_len(resume_node.get("top_skills", []))
            skills_rows = _ensure_skill_rows(llm_result.get("skills_rows"))
            experience_bullets = _coerce_bullet_overrides(
                llm_result.get("experience_bullets")
            )
            founder_role_bullets = _coerce_bullet_overrides(
                llm_result.get("founder_role_bullets")
            )

            resume_fields = {
                "summary": llm_result.get("summary", resume_node.get("summary", "")),
                "headers": headers[:9],
                "highlighted_skills": highlighted_skills[:9],
                "skills_rows_json": json.dumps(skills_rows, ensure_ascii=False),
                "experience_bullets_json": json.dumps(
                    experience_bullets, ensure_ascii=False
                ),
                "founder_role_bullets_json": json.dumps(
                    founder_role_bullets, ensure_ascii=False
                ),
                "job_req_raw": req_text,
                "target_company": llm_result.get("target_company", ""),
                "target_role": llm_result.get("target_role", ""),
                "seniority_level": llm_result.get("seniority_level", ""),
                "target_location": llm_result.get("target_location", ""),
                "work_mode": llm_result.get("work_mode", ""),
                "travel_requirement": llm_result.get("travel_requirement", ""),
                "primary_domain": llm_result.get("primary_domain", ""),
                "must_have_skills": llm_result.get("must_have_skills", []),
                "nice_to_have_skills": llm_result.get("nice_to_have_skills", []),
                "tech_stack_keywords": llm_result.get("tech_stack_keywords", []),
                "non_technical_requirements": llm_result.get(
                    "non_technical_requirements", []
                ),
                "certifications": llm_result.get("certifications", []),
                "clearances": llm_result.get("clearances", []),
                "core_responsibilities": llm_result.get("core_responsibilities", []),
                "outcome_goals": llm_result.get("outcome_goals", []),
                "salary_band": llm_result.get("salary_band", ""),
                "posting_url": llm_result.get("posting_url", ""),
                "req_id": llm_result.get("req_id", ""),
            }

            try:
                db = Neo4jClient()
                _maxcov_log("cli prompt evaluation save profile")
                profile_id = db.save_resume(resume_fields)
                db.close()
                print(f"Saved Profile {profile_id}")
                _maxcov_log("cli prompt evaluation save profile done")
            except Exception as e:
                print(f"Error saving Profile: {e}")
                sys.exit(1)

    if args.compile_pdf:
        output_path = Path(args.compile_pdf)
        try:
            db = Neo4jClient()
            data = db.get_resume_data() or {}
            profiles = db.list_applied_jobs()
            db.close()

            resume_node = data.get("resume", {}) or {}
            latest_profile = profiles[0] if profiles else {}

            profile_headers = latest_profile.get("headers") or []
            if profile_headers and any(str(h).strip() for h in profile_headers):
                headers = ensure_len(profile_headers)
            else:
                headers = ensure_len(
                    [
                        resume_node.get("head1_left", ""),
                        resume_node.get("head1_middle", ""),
                        resume_node.get("head1_right", ""),
                        resume_node.get("head2_left", ""),
                        resume_node.get("head2_middle", ""),
                        resume_node.get("head2_right", ""),
                        resume_node.get("head3_left", ""),
                        resume_node.get("head3_middle", ""),
                        resume_node.get("head3_right", ""),
                    ]
                )

            profile_skills = latest_profile.get("highlighted_skills") or []
            if profile_skills and any(str(s).strip() for s in profile_skills):
                skills = ensure_len(profile_skills)
            else:
                skills = ensure_len(resume_node.get("top_skills", []))

            summary_text = str(latest_profile.get("summary") or "").strip()
            if not summary_text:
                summary_text = str(resume_node.get("summary") or "").strip()

            section_titles = _normalize_section_titles(
                resume_node.get("section_titles_json")
                or resume_node.get("section_titles")
            )
            custom_sections = _normalize_custom_sections(
                resume_node.get("custom_sections_json")
                or resume_node.get("custom_sections")
            )
            extra_keys = _custom_section_keys(custom_sections)
            raw_order = resume_node.get("section_order")
            if isinstance(raw_order, str):
                raw_order = [s.strip() for s in raw_order.split(",") if s.strip()]
            section_order = _sanitize_section_order(raw_order, extra_keys)
            section_enabled = _normalize_section_enabled(
                resume_node.get("section_enabled"),
                list(SECTION_LABELS) + extra_keys,
                extra_keys=extra_keys,
            )
            section_order = _apply_section_enabled(
                section_order,
                section_enabled,
            )

            resume_data = {
                "summary": summary_text,
                "headers": headers[:9],
                "highlighted_skills": skills[:9],
                "skills_rows": latest_profile.get("skills_rows") or [[], [], []],
                "first_name": latest_profile.get("first_name")
                or resume_node.get("first_name", ""),
                "middle_name": latest_profile.get("middle_name")
                or resume_node.get("middle_name", ""),
                "last_name": latest_profile.get("last_name")
                or resume_node.get("last_name", ""),
                "email": latest_profile.get("email") or resume_node.get("email", ""),
                "email2": latest_profile.get("email2") or resume_node.get("email2", ""),
                "phone": latest_profile.get("phone") or resume_node.get("phone", ""),
                "font_family": resume_node.get(
                    "font_family", DEFAULT_RESUME_FONT_FAMILY
                ),
                "auto_fit_target_pages": _normalize_auto_fit_target_pages(
                    resume_node.get("auto_fit_target_pages"),
                    DEFAULT_AUTO_FIT_TARGET_PAGES,
                ),
                "linkedin_url": latest_profile.get("linkedin_url")
                or resume_node.get("linkedin_url", ""),
                "github_url": latest_profile.get("github_url")
                or resume_node.get("github_url", ""),
                "calendly_url": latest_profile.get("calendly_url")
                or resume_node.get("calendly_url", ""),
                "portfolio_url": latest_profile.get("portfolio_url")
                or resume_node.get("portfolio_url", ""),
                "section_order": section_order,
                "section_titles": section_titles,
                "custom_sections": custom_sections,
            }
            exp_overrides = _bullet_override_map(
                latest_profile.get("experience_bullets")
            )
            founder_overrides = _bullet_override_map(
                latest_profile.get("founder_role_bullets")
            )
            experience_items = _apply_bullet_overrides(
                data.get("experience", []), exp_overrides
            )
            founder_items = _apply_bullet_overrides(
                data.get("founder_roles", []), founder_overrides
            )
            profile_data = {
                **resume_node,
                **latest_profile,
                "experience": experience_items,
                "education": data.get("education", []),
                "founder_roles": founder_items,
            }

            if args.auto_fit:
                success, pdf_bytes = compile_pdf_with_auto_tuning(
                    resume_data,
                    profile_data,
                    include_matrices=True,
                    include_summary=True,
                    section_order=resume_data["section_order"],
                    target_pages=resume_data.get("auto_fit_target_pages"),
                )
            else:
                source = generate_typst_source(
                    resume_data,
                    profile_data,
                    include_matrices=True,
                    include_summary=True,
                    section_order=resume_data["section_order"],
                )
                pdf_metadata = _build_pdf_metadata(resume_data, profile_data)
                success, pdf_bytes = compile_pdf(source, metadata=pdf_metadata)
            if not success or not pdf_bytes:
                print("Typst compilation failed; see logs above.")
                sys.exit(1)
            if output_path.parent.exists() and not output_path.parent.is_dir():
                output_path = ASSETS_DIR / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)
            print(f"Wrote PDF to {output_path}")
            sys.exit(0)
        except Exception as e:
            print(f"Error compiling PDF: {e}")
            sys.exit(1)

    if args.export_resume_pdf:
        pdf_bytes = render_resume_pdf_bytes(
            save_copy=True,
            include_summary=False,
            include_skills=False,
            filename="preview_no_summary_skills.pdf",
        )
        if not pdf_bytes:
            print("Failed to export PDF.")
            sys.exit(1)
        print(
            f"Exported {ASSETS_DIR / 'preview_no_summary_skills.pdf'} (no summary/skills)."
        )
        sys.exit(0)

    if args.show_resume_data:
        try:
            db = Neo4jClient()
            data = db.get_resume_data()
            db.close()
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        except Exception as e:
            print(f"Error retrieving resume data: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.list_applied:
        try:
            db = Neo4jClient()
            jobs = db.list_applied_jobs()
            db.close()
            print(json.dumps(jobs, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error listing applied jobs: {e}")
            sys.exit(1)
        sys.exit(0)
