#!/usr/bin/env python3
import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse


ERROR_TEXTS = [
    "Unable to save to Neo4j.",
    "Database unavailable",
    "No resume found",
    "Error loading resume fields",
    "Unable to save profile bullets.",
    "PDF generation failed",
]
LLM_ERROR_HINTS = [
    "LLM call failed",
    "LLM returned",
    "Missing OpenAI API key",
    "Missing Gemini API key",
    "invalid_api_key",
]


def _log(msg: str) -> None:
    print(f"[ui-check] {msg}")


@dataclass
class UIConfig:
    base_url: str
    timeout_ms: int
    action_timeout_ms: int
    pdf_timeout_ms: int
    allow_llm_error: bool
    allow_db_error: bool
    headed: bool
    slowmo_ms: int
    screenshot_dir: str | None = None


@dataclass
class IssueRecorder:
    issues: list[str] = field(default_factory=list)

    def add(self, label: str, exc: Exception | None = None) -> None:
        if exc:
            msg = f"{label}: {type(exc).__name__}: {exc}"
        else:
            msg = label
        self.issues.append(msg)
        _log(f"issue: {msg}")

    def add_text(self, label: str, text: str) -> None:
        self.issues.append(f"{label}: {text}")
        _log(f"issue: {label}: {text}")

    def raise_if_any(self) -> None:
        if not self.issues:
            return
        _log("Failures detected:")
        for item in self.issues:
            _log(f" - {item}")
        raise SystemExit(1)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_font_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _shorten(text: str, limit: int = 280) -> str:
    cleaned = _normalize_space(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _decode_pdf_data_url(data_url: str) -> bytes:
    if not data_url.startswith("data:application/pdf"):
        raise RuntimeError("PDF embed src missing or invalid")
    payload = data_url.split(",", 1)[-1]
    if "#" in payload:
        payload = payload.split("#", 1)[0]
    if not payload:
        raise RuntimeError("PDF embed payload missing")
    return base64.b64decode(payload)


def _parse_page_count(output: str) -> int:
    count = 0
    for line in (output or "").splitlines():
        if re.match(r"^page\s+\d+", line.strip(), flags=re.IGNORECASE):
            count += 1
        if line.lstrip().startswith("<page "):
            count += 1
    if count:
        return count
    match = re.search(r"pages?\s*:\s*(\d+)", output or "", flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


@dataclass
class PDFSnapshot:
    label: str
    path: Path
    text: str
    normalized_text: str
    fonts_output: str
    pages_output: str
    page_count: int


class PDFAnalyzer:
    def __init__(
        self, recorder: IssueRecorder, cfg: UIConfig, artifact_dir: Path
    ) -> None:
        self.recorder = recorder
        self.cfg = cfg
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.tools_ready = True
        missing = [
            tool for tool in ("pdftotext", "mutool") if shutil.which(tool) is None
        ]
        if missing:
            self.recorder.add_text("missing pdf tools", ", ".join(missing))
            self.tools_ready = False

    def _safe_label(self, label: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip() or "pdf")
        return cleaned[:80] or "pdf"

    def _run_cmd(self, cmd: list[str], label: str) -> str:
        timeout_s = max(5.0, self.cfg.pdf_timeout_ms / 1000.0)
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except Exception as exc:
            self.recorder.add(label, exc)
            return ""
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            msg = stderr or stdout or f"exit {result.returncode}"
            self.recorder.add_text(label, _shorten(msg))
        return stdout or stderr

    def analyze_bytes(self, label: str, pdf_bytes: bytes) -> PDFSnapshot:
        safe = self._safe_label(label)
        path = self.artifact_dir / f"{safe}.pdf"
        path.write_bytes(pdf_bytes)
        if not self.tools_ready:
            return PDFSnapshot(
                label=label,
                path=path,
                text="",
                normalized_text="",
                fonts_output="",
                pages_output="",
                page_count=0,
            )
        text = self._run_cmd(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
            f"pdftotext {label}",
        )
        fonts_output = self._run_cmd(
            ["mutool", "info", "-F", str(path)], f"mutool info {label}"
        )
        pages_output = self._run_cmd(
            ["mutool", "pages", str(path)], f"mutool pages {label}"
        )
        page_count = _parse_page_count(pages_output)
        return PDFSnapshot(
            label=label,
            path=path,
            text=text,
            normalized_text=_normalize_space(text).lower(),
            fonts_output=fonts_output,
            pages_output=pages_output,
            page_count=page_count,
        )


class ResumeUI:
    def __init__(self, page, recorder: IssueRecorder, cfg: UIConfig) -> None:
        self.page = page
        self.recorder = recorder
        self.cfg = cfg
        self.last_custom_section_title: str | None = None
        self._state_update_counter = 0
        self.page.on("websocket", self._on_websocket)

    def _on_websocket(self, ws) -> None:
        ws.on("framereceived", self._on_ws_frame)

    def _on_ws_frame(self, payload) -> None:
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="ignore")
            if not payload:
                return
            if "delta" in payload:
                self._state_update_counter += 1
        except Exception:
            return

    def state_update_token(self) -> int:
        return self._state_update_counter

    def wait_for_state_update(self, token: int, timeout_ms: int | None = None) -> bool:
        timeout_s = (timeout_ms or self.cfg.timeout_ms) / 1000.0
        deadline = time.time() + max(1.0, timeout_s)
        while time.time() < deadline:
            if self._state_update_counter > token:
                return True
            time.sleep(0.05)
        return False

    def left_panel(self):
        heading = self.page.get_by_role("heading", name="Resume Builder")
        if heading.count() > 0:
            container = heading.first.locator("..")
            for _ in range(12):
                if container.count() == 0:
                    break
                try:
                    scrollable = container.first.evaluate(
                        """
                        (el) => {
                          const style = window.getComputedStyle(el);
                          const val = (style.overflowY || style.overflow || "").toLowerCase();
                          return val.includes("auto") || val.includes("scroll");
                        }
                        """
                    )
                except Exception:
                    scrollable = False
                if scrollable:
                    return container.first
                container = container.locator("..")
        panel = self.page.locator(
            'div[style*="overflow:auto"], div[style*="overflow: auto"], '
            'div[style*="overflow-y:auto"], div[style*="overflow-y: auto"]'
        ).first
        if panel.count() > 0:
            return panel
        raise RuntimeError("Left panel not found")

    def section_block(
        self, heading: str, add_button: str, field_placeholder: str | None = None
    ):
        panel = self.left_panel()
        header = panel.get_by_role("heading", name=heading)
        container = header.locator("..")
        for _ in range(8):
            if container.get_by_role("button", name=add_button).count() > 0:
                if field_placeholder:
                    if container.get_by_placeholder(field_placeholder).count() == 0:
                        container = container.locator("..")
                        continue
                return container
            container = container.locator("..")
        return panel

    def _add_section_container(self):
        panel = self.left_panel()
        label = panel.get_by_text("Add section", exact=True)
        container = label.locator("..")
        for _ in range(6):
            if (
                container.get_by_role("button", name="Add Section").count() > 0
                and container.get_by_placeholder("Section title").count() > 0
            ):
                return container
            container = container.locator("..")
        return panel

    def goto(self) -> None:
        self.page.goto(self.cfg.base_url, wait_until="domcontentloaded")
        self.wait_for_app_ready()

    def wait_for_app_ready(self) -> None:
        loading = self.page.get_by_text("Loading Resume Builder", exact=False)
        if loading.count() > 0:
            loading.first.wait_for(state="hidden", timeout=self.cfg.timeout_ms)
        self.expect_heading("Resume Builder")
        self.page.get_by_placeholder(
            "Paste or type the job requisition here"
        ).first.wait_for(timeout=self.cfg.timeout_ms)

    def expect_heading(self, text: str) -> None:
        self.page.get_by_role("heading", name=text).wait_for(
            timeout=self.cfg.timeout_ms
        )

    def expect_text(self, text: str) -> None:
        self.page.get_by_text(text, exact=False).first.wait_for(
            timeout=self.cfg.timeout_ms
        )

    def scroll_to_text(self, text: str) -> None:
        panel = self.left_panel()
        panel.get_by_text(text, exact=False).first.scroll_into_view_if_needed()

    def fill_placeholder(self, placeholder: str, value: str) -> None:
        field = self.left_panel().get_by_placeholder(placeholder)
        field.first.fill(value, timeout=self.cfg.action_timeout_ms)

    def fill_placeholder_at(self, placeholder: str, index: int, value: str) -> None:
        field = self.left_panel().get_by_placeholder(placeholder).nth(index)
        field.fill(value, timeout=self.cfg.action_timeout_ms)

    def fill_textarea(self, placeholder: str, value: str) -> None:
        area = self.left_panel().get_by_placeholder(placeholder)
        area.first.fill(value, timeout=self.cfg.action_timeout_ms)

    def fill_textarea_at(self, placeholder: str, index: int, value: str) -> None:
        area = self.left_panel().get_by_placeholder(placeholder).nth(index)
        area.fill(value, timeout=self.cfg.action_timeout_ms)

    def fill_summary(self, value: str) -> None:
        panel = self.left_panel()
        label = panel.get_by_text("Professional Summary", exact=True)
        if label.count() == 0:
            label = panel.get_by_text("Professional Summary", exact=False)
        if label.count() == 0:
            raise RuntimeError("Professional Summary label not found")
        label.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        area = label.first.locator("xpath=following::textarea[1]")
        if area.count() == 0:
            raise RuntimeError("Professional Summary textarea not found")
        area.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        area.first.fill(value, timeout=self.cfg.action_timeout_ms)

    def click_button(self, name: str) -> None:
        btn = self.page.get_by_role("button", name=name)
        if btn.count() == 0:
            raise RuntimeError(f"Button not found: {name}")
        btn.first.click(timeout=self.cfg.action_timeout_ms)

    def select_first_model(self) -> None:
        combo = self.page.get_by_role("combobox")
        if combo.count() > 0:
            combo.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
            combo.first.click(timeout=self.cfg.action_timeout_ms)
            options = self.page.locator('[role="option"]')
            if options.count() == 0:
                raise RuntimeError("Model options not found")
            options.first.click(timeout=self.cfg.action_timeout_ms)
            return
        select = self.page.locator("select").first
        select.wait_for(state="attached", timeout=self.cfg.timeout_ms)
        select.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        options = select.locator("option")
        if options.count() == 0:
            raise RuntimeError("Model options not found")
        value = options.nth(0).get_attribute("value")
        if value:
            select.select_option(value)

    def set_clipboard(self, text: str) -> None:
        self.page.evaluate(
            """
            async (payload) => {
              if (!navigator.clipboard) return false;
              await navigator.clipboard.writeText(payload);
              return true;
            }
            """,
            text,
        )

    def find_switch_by_label(self, label: str):
        label_loc = self.left_panel().get_by_text(label, exact=False).first
        container = label_loc.locator("..")
        switch = container.get_by_role("switch")
        if switch.count() == 0:
            container = container.locator("..")
            switch = container.get_by_role("switch")
        if switch.count() == 0:
            raise RuntimeError(f"Switch not found for label: {label}")
        return switch.first

    def set_switch(self, label: str, value: bool) -> None:
        switch = self.find_switch_by_label(label)
        checked = switch.is_checked()
        if checked != value:
            token = self.state_update_token()
            switch.click(timeout=self.cfg.action_timeout_ms)
            deadline = time.time() + (self.cfg.action_timeout_ms / 1000.0)
            while time.time() < deadline:
                if switch.is_checked() == value:
                    break
                time.sleep(0.05)
            self.wait_for_state_update(token, self.cfg.action_timeout_ms)

    def toggle_switch(self, label: str) -> None:
        switch = self.find_switch_by_label(label)
        switch.click(timeout=self.cfg.action_timeout_ms)

    def wait_for_pdf(self) -> None:
        embed = self.page.locator('embed[type="application/pdf"]')
        embed.wait_for(timeout=self.cfg.pdf_timeout_ms)
        src = embed.first.get_attribute("src") or ""
        if not src.startswith("data:application/pdf"):
            raise RuntimeError("PDF embed src missing")

    def wait_for_pipeline_done(self) -> None:
        tokens = [
            "Auto-pipeline complete",
            "Auto-pipeline failed",
            "Stage 1 failed",
            "Stage 2 failed",
            "Stage 3 failed",
            "Auto-pipeline skipped",
        ]
        self.page.wait_for_function(
            "(tokens) => { const body = document.body; if (!body) return false; "
            "const text = body.innerText || ''; return tokens.some(t => text.includes(t)); }",
            arg=tokens,
            timeout=max(self.cfg.timeout_ms, 30000),
        )

    def current_pdf_src(self) -> str | None:
        embed = self.page.locator('embed[type="application/pdf"]')
        if embed.count() == 0:
            return None
        src = embed.first.get_attribute("src") or ""
        if not src.startswith("data:application/pdf"):
            return None
        return src

    def pdf_data_url(self, previous_src: str | None = None) -> str:
        embed = self.page.locator('embed[type="application/pdf"]')
        embed.wait_for(timeout=self.cfg.pdf_timeout_ms)
        deadline = time.time() + (self.cfg.pdf_timeout_ms / 1000.0)
        while time.time() < deadline:
            src = embed.first.get_attribute("src") or ""
            if src.startswith("data:application/pdf"):
                if previous_src and src == previous_src:
                    time.sleep(0.2)
                    continue
                return src
            time.sleep(0.2)
        if previous_src:
            raise RuntimeError("PDF embed src unchanged")
        raise RuntimeError("PDF embed src missing")

    def get_pdf_bytes(self, previous_src: str | None = None) -> bytes:
        return _decode_pdf_data_url(self.pdf_data_url(previous_src=previous_src))

    def wait_for_data_loaded(self) -> None:
        self.left_panel()
        deadline = time.time() + (self.cfg.timeout_ms / 1000.0)
        while time.time() < deadline:
            if (
                self.left_panel()
                .get_by_role("heading", name="Contact Info")
                .count()
                > 0
            ):
                return
            if self.left_panel().get_by_text("Data not loaded", exact=False).count() > 0:
                time.sleep(1.0)
            else:
                time.sleep(0.5)
        raise RuntimeError("Timed out waiting for Contact Info")

    def read_error_texts(self, allow_llm: bool, allow_db: bool) -> None:
        for text in ERROR_TEXTS:
            if self.page.get_by_text(text, exact=False).count() > 0:
                if allow_db and ("Neo4j" in text or "Database" in text):
                    _log(f"warning: {text}")
                    continue
                raise RuntimeError(text)
        for text in LLM_ERROR_HINTS:
            if self.page.get_by_text(text, exact=False).count() > 0:
                if allow_llm:
                    _log(f"warning: {text}")
                    return
                raise RuntimeError(text)

    def section_row_count(self) -> int:
        return self.left_panel().get_by_label("Toggle section visibility").count()

    def section_title_input(self, idx: int):
        return self.left_panel().get_by_placeholder("Section title").nth(idx)

    def section_titles(self) -> list[str]:
        titles = []
        count = self.section_row_count()
        for idx in range(count):
            try:
                field = self.section_title_input(idx)
                field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
                titles.append(field.input_value().strip())
            except Exception:
                titles.append("")
        return titles

    def find_section_row_by_title(self, title: str):
        panel = self.left_panel()
        inputs = panel.get_by_placeholder("Section title")
        target = title.strip().lower()
        for idx in range(inputs.count()):
            try:
                field = inputs.nth(idx)
                value = field.input_value().strip().lower()
            except Exception:
                continue
            if not value:
                continue
            if value == target or target in value:
                row = field.locator(
                    "xpath=ancestor::*[.//*[@aria-label='Toggle section visibility'] and "
                    ".//*[@aria-label='Move section up']][1]"
                )
                if row.count() > 0:
                    return row
        return None

    def set_section_title(self, idx: int, value: str) -> None:
        self.section_title_input(idx).fill(value, timeout=self.cfg.action_timeout_ms)

    def toggle_section_visibility(self, idx: int) -> None:
        checkbox = self.left_panel().get_by_label("Toggle section visibility").nth(idx)
        checkbox.click(timeout=self.cfg.action_timeout_ms)

    def set_section_visibility(self, idx: int, value: bool) -> None:
        checkbox = self.left_panel().get_by_label("Toggle section visibility").nth(idx)
        checked = checkbox.is_checked()
        if checked != value:
            token = self.state_update_token()
            checkbox.click(timeout=self.cfg.action_timeout_ms)
            deadline = time.time() + (self.cfg.action_timeout_ms / 1000.0)
            while time.time() < deadline:
                if checkbox.is_checked() == value:
                    break
                time.sleep(0.05)
            self.wait_for_state_update(token, self.cfg.action_timeout_ms)

    def move_section(self, idx: int, direction: str) -> None:
        if direction == "up":
            self.left_panel().get_by_label("Move section up").nth(idx).click(
                timeout=self.cfg.action_timeout_ms
            )
        else:
            self.left_panel().get_by_label("Move section down").nth(idx).click(
                timeout=self.cfg.action_timeout_ms
            )

    def set_font_family(self, value: str) -> None:
        ok = self.page.evaluate(
            """
            (family) => {
              const input = document.getElementById("resume-font-picker");
              if (!input) return false;
              input.value = family;
              input.dispatchEvent(new Event("input", { bubbles: true }));
              input.dispatchEvent(new Event("change", { bubbles: true }));
              return true;
            }
            """,
            value,
        )
        if ok:
            return
        button = self.page.locator(
            ".fp__button, .fp__btn, button:has-text(\"Pick a font\")"
        )
        if button.count() > 0:
            button.first.click(timeout=self.cfg.action_timeout_ms)
            search = self.page.locator(
                ".fp__search input, input.fp__search, input[type=\"search\"]"
            )
            if search.count() > 0:
                search.first.fill(value, timeout=self.cfg.action_timeout_ms)
            option = self.page.locator(
                ".fp__list .fp__item, .fp__option, .fp__list li"
            )
            if option.count() > 0:
                option.first.click(timeout=self.cfg.action_timeout_ms)
                return
        raise RuntimeError("Font input not found")

    def set_auto_fit_pages(self, value: str) -> None:
        row = self.page.get_by_text("Auto-fit", exact=False).first
        container = row.locator("..")
        number_input = container.locator('input[type="number"]')
        if number_input.count() == 0:
            container = container.locator("..")
            number_input = container.locator('input[type="number"]')
        if number_input.count() == 0:
            raise RuntimeError("Auto-fit page input not found")
        number_input.first.fill(value, timeout=self.cfg.action_timeout_ms)

    def find_prompt_editor(self):
        return self.page.locator("#prompt-yaml-field")

    def fill_prompt_editor(self, value: str) -> None:
        ok = self.page.evaluate(
            """
            (text) => {
              const textarea = document.getElementById("prompt-yaml-field");
              if (!textarea) return false;
              if (window.__promptYamlCodeMirror) {
                window.__promptYamlCodeMirror.setValue(text);
                window.__promptYamlCodeMirror.save();
                textarea.dispatchEvent(new Event("input", { bubbles: true }));
                return true;
              }
              textarea.value = text;
              textarea.dispatchEvent(new Event("input", { bubbles: true }));
              return true;
            }
            """,
            value,
        )
        if not ok:
            raise RuntimeError("Prompt editor not found")

    def add_custom_section(self, title: str, body: str) -> None:
        self.scroll_to_text("Section order")
        panel = self.left_panel()
        try:
            panel.get_by_text("Add section", exact=True).first.scroll_into_view_if_needed(
                timeout=self.cfg.timeout_ms
            )
        except Exception:
            pass
        form = self._add_section_container()
        add_title = form.get_by_placeholder("Section title").first
        add_body = form.get_by_placeholder(
            "Section content (one bullet per line)"
        ).first
        add_button = form.get_by_role("button", name="Add Section").first
        cards = panel.locator('[data-custom-section="1"]')
        before = cards.count()
        add_title.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        add_title.fill(title, timeout=self.cfg.action_timeout_ms)
        add_body.fill(body, timeout=self.cfg.action_timeout_ms)
        add_button.click(timeout=self.cfg.action_timeout_ms)
        self.last_custom_section_title = title
        deadline = time.time() + (self.cfg.timeout_ms / 1000.0)
        while time.time() < deadline:
            if cards.count() > before:
                return
            time.sleep(0.2)
        raise RuntimeError("Custom section did not appear after add")

    def remove_custom_section(self) -> None:
        panel = self.left_panel()
        cards = panel.locator('[data-custom-section="1"]')
        if cards.count() == 0:
            raise RuntimeError("Custom section card not found")
        card = cards.last
        btn = card.get_by_role("button", name="Remove")
        if btn.count() > 0:
            btn.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
            btn.first.click(timeout=self.cfg.action_timeout_ms)
            return
        remove_btns = panel.get_by_role("button", name="Remove")
        if remove_btns.count() == 0:
            raise RuntimeError("Custom section remove button not found")
        remove_btns.last.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        remove_btns.last.click(timeout=self.cfg.action_timeout_ms)

    def _section_container(self):
        header = self.left_panel().get_by_text("Section order", exact=True)
        container = header.locator("..")
        for _ in range(6):
            if container.get_by_role("button", name="Add Section").count() > 0:
                return container
            container = container.locator("..")
        return container


class UITestRunner:
    def __init__(self, ui: ResumeUI, recorder: IssueRecorder, cfg: UIConfig) -> None:
        self.ui = ui
        self.recorder = recorder
        self.cfg = cfg
        self.data_loaded = False
        self.custom_section_added = False
        self.artifact_dir = (
            Path(cfg.screenshot_dir)
            if cfg.screenshot_dir
            else Path(tempfile.mkdtemp(prefix="ui_check_artifacts_"))
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_analyzer = PDFAnalyzer(recorder, cfg, self.artifact_dir)
        self.font_family = "Avenir LT Std"
        self.summary_head = "QA SUMMARY HEAD."
        self.summary_tail = "QA SUMMARY TAIL."
        self.summary_text = f"{self.summary_head} {self.summary_tail}"
        self.contact_updates = {
            "First Name": "Test",
            "Middle Name / Initial": "Q",
            "Last Name": "User",
            "Email": "test@example.com",
            "Secondary Email": "test2@example.com",
            "Phone": "555-555-5555",
            "LinkedIn URL": "linkedin.com/in/test-user",
            "GitHub URL": "git.new/test-user",
            "Google Scholar URL": "https://scholar.google.com/citations?user=TEST",
            "Calendly URL": "https://cal.link/test",
            "Portfolio URL": "https://example.com/portfolio",
        }
        self.experience_entry = {
            "role": "QA Lead",
            "company": "ExampleCo",
            "location": "Remote",
            "description": "Led test automation and release reliability.",
            "bullets": ["Did thing A", "Did thing B"],
        }
        self.education_entry = {
            "degree": "M.S.",
            "school": "Test University",
            "location": "Test City",
            "description": "Focus on systems and ML.",
            "bullets": ["Course A", "Course B"],
        }
        self.founder_entry = {
            "role": "Founder",
            "company": "Test Startup",
            "location": "Remote",
            "description": "Built a prototype and shipped MVP.",
            "bullets": ["Milestone A", "Milestone B"],
        }
        self.custom_section_title = "QA Custom Section"
        self.custom_section_bullets = ["Custom Bullet One", "Custom Bullet Two"]
        self.section_title_overrides = {
            "Skills": "QA Skills Section",
            "Education": "QA Education Section",
            "Experience": "QA Experience Section",
            "Startup Founder": "QA Founder Section",
        }

    def _step(self, name: str, fn: Callable[[], None]) -> None:
        _log(f"step: {name}")
        try:
            fn()
        except Exception as exc:
            self._capture_failure_artifact(name)
            self.recorder.add(name, exc)

    def _capture_failure_artifact(self, name: str) -> None:
        try:
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip() or "step")
            stamp = int(time.time() * 1000)
            path = self.artifact_dir / f"failure_{safe}_{stamp}.png"
            self.ui.page.screenshot(path=str(path), full_page=True)
            _log(f"saved screenshot: {path}")
        except Exception:
            return

    def _scan_errors(self) -> None:
        self.ui.read_error_texts(
            allow_llm=self.cfg.allow_llm_error,
            allow_db=self.cfg.allow_db_error,
        )

    def _find_section_index(self, title: str) -> int:
        titles = self.ui.section_titles()
        lowered = title.strip().lower()
        for idx, value in enumerate(titles):
            if value.strip().lower() == lowered:
                return idx
        for idx, value in enumerate(titles):
            if lowered in value.strip().lower():
                return idx
        raise RuntimeError(f"Section title not found: {title}")

    def _set_section_visibility_by_title(self, title: str, value: bool) -> None:
        try:
            idx = self._find_section_index(title)
            self.ui.section_title_input(idx).scroll_into_view_if_needed(
                timeout=self.cfg.timeout_ms
            )
            self.ui.set_section_visibility(idx, value)
            return
        except Exception:
            pass
        try:
            row = self.ui.find_section_row_by_title(title)
            if row is None:
                raise RuntimeError("row not found")
            checkbox = row.get_by_label("Toggle section visibility").first
            checked = checkbox.is_checked()
            if checked != value:
                token = self.ui.state_update_token()
                checkbox.click(timeout=self.cfg.action_timeout_ms)
                deadline = time.time() + (self.cfg.action_timeout_ms / 1000.0)
                while time.time() < deadline:
                    if checkbox.is_checked() == value:
                        break
                    time.sleep(0.05)
                self.ui.wait_for_state_update(token, self.cfg.action_timeout_ms)
        except Exception:
            self.recorder.add_text("section visibility missing", title)

    def _move_section_before(self, title: str, before_title: str) -> None:
        try:
            for _ in range(25):
                idx = self._find_section_index(title)
                target_idx = self._find_section_index(before_title)
                if idx <= target_idx:
                    return
                row = self.ui.find_section_row_by_title(title)
                if row is None:
                    break
                btn = row.get_by_label("Move section up").first
                btn.click(timeout=self.cfg.action_timeout_ms)
        except Exception:
            self.recorder.add_text(
                "section reorder missing", f"{title} before {before_title}"
            )
            return
        self.recorder.add_text(
            "section reorder failed", f"{title} before {before_title}"
        )

    def _generate_pdf_snapshot(self, label: str) -> PDFSnapshot:
        self.ui.page.wait_for_timeout(400)
        self.ui.fill_placeholder("Paste or type the job requisition here", "")
        previous_src = self.ui.current_pdf_src()
        self.ui.click_button("Generate PDF")
        pdf_bytes = self.ui.get_pdf_bytes(previous_src=previous_src)
        return self.pdf_analyzer.analyze_bytes(label, pdf_bytes)

    def _assert_pdf_contains(self, snapshot: PDFSnapshot, tokens: Iterable[str], ctx: str) -> None:
        text = snapshot.normalized_text
        for token in tokens:
            normalized = _normalize_space(token).lower()
            if normalized and normalized not in text:
                self.recorder.add_text(f"{ctx} missing text", token)

    def _assert_pdf_missing(self, snapshot: PDFSnapshot, tokens: Iterable[str], ctx: str) -> None:
        text = snapshot.normalized_text
        for token in tokens:
            normalized = _normalize_space(token).lower()
            if normalized and normalized in text:
                self.recorder.add_text(f"{ctx} unexpected text", token)

    def _assert_pdf_order(self, snapshot: PDFSnapshot, tokens: Iterable[str], ctx: str) -> None:
        text = snapshot.normalized_text
        positions = []
        for token in tokens:
            normalized = _normalize_space(token).lower()
            if not normalized:
                continue
            idx = text.find(normalized)
            if idx < 0:
                self.recorder.add_text(f"{ctx} missing text", token)
                return
            positions.append(idx)
        if positions != sorted(positions):
            self.recorder.add_text(f"{ctx} order", " -> ".join(tokens))

    def _assert_pdf_font(self, snapshot: PDFSnapshot, font_name: str) -> None:
        if not snapshot.fonts_output:
            self.recorder.add_text("pdf font output missing", snapshot.label)
            return
        if _normalize_font_token(font_name) not in _normalize_font_token(
            snapshot.fonts_output
        ):
            self.recorder.add_text("pdf font missing", font_name)

    def _assert_pdf_pages(
        self,
        snapshot: PDFSnapshot,
        *,
        equals: int | None = None,
        at_least: int | None = None,
        at_most: int | None = None,
        check_letter: bool = False,
    ) -> None:
        count = snapshot.page_count
        if count <= 0:
            self.recorder.add_text("pdf pages missing", snapshot.label)
            return
        if equals is not None and count != equals:
            self.recorder.add_text("pdf page count", f"{snapshot.label}: {count}")
        if at_least is not None and count < at_least:
            self.recorder.add_text("pdf page count", f"{snapshot.label}: {count}")
        if at_most is not None and count > at_most:
            self.recorder.add_text("pdf page count", f"{snapshot.label}: {count}")
        if check_letter and snapshot.pages_output:
            if not (
                re.search(r"612\s*x\s*792", snapshot.pages_output)
                or re.search(r'r="612"\s+t="792"', snapshot.pages_output)
                or ("612" in snapshot.pages_output and "792" in snapshot.pages_output)
            ):
                self.recorder.add_text("pdf page size", snapshot.label)

    def run(self) -> None:
        self._step("open app", self.ui.goto)
        self._step("select model", self.ui.select_first_model)
        self._step(
            "fill job req",
            lambda: self.ui.fill_placeholder(
                "Paste or type the job requisition here",
                "Playwright UI test job req.",
            ),
        )
        self._step(
            "prompt editor",
            lambda: self.ui.fill_prompt_editor("name: ui_check\nversion: 1\n"),
        )
        self._step(
            "toggle include skills",
            lambda: self.ui.toggle_switch("Include Skills section"),
        )
        self._step(
            "toggle include skills back",
            lambda: self.ui.toggle_switch("Include Skills section"),
        )
        self._step(
            "toggle rewrite bullets",
            lambda: self.ui.toggle_switch("Rewrite bullets with LLM"),
        )
        self._step("auto-fit toggle", lambda: self.ui.toggle_switch("Auto-fit"))
        self._step("auto-fit toggle back", lambda: self.ui.toggle_switch("Auto-fit"))
        self._step("auto-fit pages", lambda: self.ui.set_auto_fit_pages("3"))
        self._step("auto-fit pages reset", lambda: self.ui.set_auto_fit_pages("2"))
        self._step("set font", lambda: self.ui.set_font_family(self.font_family))
        self._step("section order checks", self._section_order_checks)
        self._step("load data", lambda: self.ui.click_button("Load Data"))
        self._step("wait data loaded", self._ensure_data_loaded)
        if self.data_loaded:
            self._step("fill contact info", self._fill_contact_info)
            self._step("fill summary", self._fill_summary)
            self._step("add custom section", self._add_custom_section)
            self._step("experience form", self._fill_experience)
            self._step("education form", self._fill_education)
            self._step("founder form", self._fill_founder)
            self._step("toggle profile bullets edit", self._toggle_profile_bullets)
            self._step("fill profile bullets", self._fill_profile_bullets)
            self._step(
                "save profile bullets",
                lambda: self.ui.click_button("Save Profile Bullets"),
            )
            self._step("save data", lambda: self.ui.click_button("Save Data"))
            self._step("configure section titles", self._configure_section_titles)
            self._step("reorder sections", self._reorder_sections)
            self._step("pdf baseline", self._pdf_baseline_checks)
            self._step("pdf skills hidden", self._pdf_without_skills)
            self._step("pdf education hidden", self._pdf_without_education)
            self._step("pdf custom removed", self._pdf_without_custom_section)
            self._step("pdf auto-fit one page", self._pdf_auto_fit_one_page)
        else:
            self.recorder.add_text("data load", "skipping form steps")
        self._step("clipboard paste", self._paste_job_req)
        self._step("wait pipeline from paste", self._wait_pipeline_done)
        self._step(
            "job req for pipeline",
            lambda: self.ui.fill_placeholder(
                "Paste or type the job requisition here",
                "Pipeline job req text.",
            ),
        )
        self._step("run pipeline", lambda: self.ui.click_button("Run Pipeline"))
        self._step("wait pipeline", self._wait_pipeline_done)
        self._step("scan errors", self._scan_errors)

    def _ensure_data_loaded(self) -> None:
        try:
            self.ui.wait_for_data_loaded()
            self.data_loaded = True
        except Exception as exc:
            self.data_loaded = False
            raise exc

    def _paste_job_req(self) -> None:
        self.ui.set_clipboard("Paste flow text.")
        self.ui.click_button("Paste & Run")

    def _wait_pipeline_done(self) -> None:
        try:
            self.ui.wait_for_pipeline_done()
        except Exception as exc:
            if self.cfg.allow_llm_error or self.cfg.allow_db_error:
                _log(f"warning: pipeline wait timed out: {exc}")
                return
            raise

    def _add_custom_section(self) -> None:
        body = "\n".join(f"- {line}" for line in self.custom_section_bullets)
        try:
            self.ui.add_custom_section(self.custom_section_title, body)
            self.custom_section_added = True
            return
        except Exception:
            try:
                titles = [t.strip().lower() for t in self.ui.section_titles()]
            except Exception:
                titles = []
            if self.custom_section_title.strip().lower() in titles:
                self.custom_section_added = True
                return
            raise

    def _section_order_checks(self) -> None:
        self.ui.scroll_to_text("Section order")
        count = self.ui.section_row_count()
        if count == 0:
            raise RuntimeError("No section rows found")
        first_field = self.ui.section_title_input(0)
        first_field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        first_title = first_field.input_value()
        self.ui.toggle_section_visibility(0)
        self.ui.toggle_section_visibility(0)
        if count > 1:
            self.ui.move_section(0, "down")
            self.ui.move_section(1, "up")
        updated_field = self.ui.section_title_input(0)
        updated_field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        updated_title = updated_field.input_value()
        if not updated_title:
            self.ui.section_title_input(0).fill(first_title or "Summary")

    def _fill_summary(self) -> None:
        self.ui.fill_summary(self.summary_text)

    def _fill_contact_info(self) -> None:
        self.ui.scroll_to_text("Contact Info")
        for placeholder, value in self.contact_updates.items():
            self.ui.fill_placeholder(placeholder, value)

    def _fill_experience(self) -> None:
        block = self.ui.section_block("Experience", "Add Experience", "Role")
        btn = block.get_by_role("button", name="Add Experience")
        btn.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        btn.click(timeout=self.cfg.action_timeout_ms)
        role_field = block.get_by_placeholder("Role").last
        role_field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        role_field.fill(self.experience_entry["role"], timeout=self.cfg.action_timeout_ms)
        block.get_by_placeholder("Company").last.fill(
            self.experience_entry["company"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder("Location").last.fill(
            self.experience_entry["location"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder("Short company/role description (optional)").last.fill(
            self.experience_entry["description"],
            timeout=self.cfg.action_timeout_ms,
        )
        block.get_by_placeholder("Bullets (one per line)").last.fill(
            "\n".join(self.experience_entry["bullets"]),
            timeout=self.cfg.action_timeout_ms,
        )

    def _fill_education(self) -> None:
        block = self.ui.section_block("Education", "Add Education", "Degree")
        btn = block.get_by_role("button", name="Add Education")
        btn.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        btn.click(timeout=self.cfg.action_timeout_ms)
        degree_field = block.get_by_placeholder("Degree").last
        degree_field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        degree_field.fill(self.education_entry["degree"], timeout=self.cfg.action_timeout_ms)
        block.get_by_placeholder("School").last.fill(
            self.education_entry["school"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder("Location").last.fill(
            self.education_entry["location"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder(
            "Program description or highlights (optional)"
        ).last.fill(self.education_entry["description"], timeout=self.cfg.action_timeout_ms)
        block.get_by_placeholder("Bullets (one per line)").last.fill(
            "\n".join(self.education_entry["bullets"]),
            timeout=self.cfg.action_timeout_ms,
        )

    def _fill_founder(self) -> None:
        block = self.ui.section_block("Founder Roles", "Add Founder Role", "Role")
        btn = block.get_by_role("button", name="Add Founder Role")
        btn.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        btn.click(timeout=self.cfg.action_timeout_ms)
        role_field = block.get_by_placeholder("Role").last
        role_field.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        role_field.fill(self.founder_entry["role"], timeout=self.cfg.action_timeout_ms)
        block.get_by_placeholder("Company").last.fill(
            self.founder_entry["company"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder("Location").last.fill(
            self.founder_entry["location"], timeout=self.cfg.action_timeout_ms
        )
        block.get_by_placeholder("Company or role description (optional)").last.fill(
            self.founder_entry["description"],
            timeout=self.cfg.action_timeout_ms,
        )
        block.get_by_placeholder("Bullets (one per line)").last.fill(
            "\n".join(self.founder_entry["bullets"]),
            timeout=self.cfg.action_timeout_ms,
        )

    def _toggle_profile_bullets(self) -> None:
        block = self.ui.section_block("Experience", "Add Experience")
        label = block.get_by_text("Edit LLM bullets", exact=False)
        if label.count() == 0:
            raise RuntimeError("Edit LLM bullets toggle not found")
        label.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        toggle = label.locator("..").get_by_role("switch")
        if toggle.count() == 0:
            toggle = block.get_by_role("switch").first
        if toggle.count() == 0:
            raise RuntimeError("Edit LLM bullets switch not found")
        toggle.first.click(timeout=self.cfg.action_timeout_ms)

    def _fill_profile_bullets(self) -> None:
        block = self.ui.section_block(
            "Experience", "Add Experience", "Profile bullets (one per line)"
        )
        areas = block.get_by_placeholder("Profile bullets (one per line)")
        if areas.count() == 0:
            self.recorder.add_text("profile bullets", "inputs not found; skipping")
            return
        areas.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
        areas.first.fill("Profile bullet A", timeout=self.cfg.action_timeout_ms)
        if areas.count() > 1:
            areas.last.fill("Profile bullet B", timeout=self.cfg.action_timeout_ms)

    def _configure_section_titles(self) -> None:
        self.ui.scroll_to_text("Section order")
        for base_title, new_title in self.section_title_overrides.items():
            try:
                row = self.ui.find_section_row_by_title(base_title)
                if row is None:
                    row = self.ui.find_section_row_by_title(new_title)
                if row is None:
                    raise RuntimeError("row not found")
                row.get_by_placeholder("Section title").first.fill(
                    new_title, timeout=self.cfg.action_timeout_ms
                )
            except Exception:
                self.recorder.add_text("section title missing", base_title)
                continue
        if self.custom_section_added:
            try:
                row = self.ui.find_section_row_by_title(self.custom_section_title)
                if row is None:
                    raise RuntimeError("row not found")
                row.get_by_placeholder("Section title").first.fill(
                    self.custom_section_title, timeout=self.cfg.action_timeout_ms
                )
            except Exception:
                self.recorder.add_text(
                    "section title missing", self.custom_section_title
                )

    def _reorder_sections(self) -> None:
        self.ui.scroll_to_text("Section order")
        self._move_section_before(
            self.section_title_overrides["Experience"],
            self.section_title_overrides["Education"],
        )

    def _pdf_baseline_checks(self) -> None:
        self.ui.set_switch("Include Skills section", True)
        self.ui.set_switch("Auto-fit", True)
        self.ui.set_auto_fit_pages("2")
        self.ui.set_font_family(self.font_family)
        snapshot = self._generate_pdf_snapshot("pdf_baseline")
        self._assert_pdf_contains(
            snapshot,
            [
                self.contact_updates["Email"],
                self.contact_updates["Phone"],
                "portfolio",
                self.summary_head,
                self.summary_tail,
                "Profile bullet A",
                self.education_entry["degree"],
                self.education_entry["school"],
                "Profile bullet B",
                self.section_title_overrides["Skills"],
                self.section_title_overrides["Education"],
                self.section_title_overrides["Experience"],
                self.section_title_overrides["Startup Founder"],
            ],
            "pdf baseline",
        )
        self._assert_pdf_order(
            snapshot,
            [
                self.section_title_overrides["Experience"],
                self.section_title_overrides["Education"],
            ],
            "pdf baseline",
        )
        self._assert_pdf_font(snapshot, "Avenir")
        self._assert_pdf_pages(snapshot, at_least=1, check_letter=True)

    def _pdf_without_skills(self) -> None:
        self.ui.scroll_to_text("Include Skills section")
        self.ui.set_switch("Include Skills section", False)
        self._set_section_visibility_by_title(
            self.section_title_overrides["Skills"], False
        )
        snapshot = self._generate_pdf_snapshot("pdf_no_skills")
        self._assert_pdf_missing(
            snapshot, [self.section_title_overrides["Skills"]], "skills toggle"
        )
        self._assert_pdf_contains(
            snapshot, [self.section_title_overrides["Experience"]], "skills toggle"
        )

    def _pdf_without_education(self) -> None:
        self._set_section_visibility_by_title(
            self.section_title_overrides["Education"], False
        )
        self.ui.page.wait_for_timeout(600)
        snapshot = self._generate_pdf_snapshot("pdf_no_education")
        self._assert_pdf_missing(
            snapshot, [self.section_title_overrides["Education"]], "education toggle"
        )
        self._assert_pdf_contains(
            snapshot, [self.section_title_overrides["Experience"]], "education toggle"
        )

    def _pdf_without_custom_section(self) -> None:
        if self.custom_section_added:
            try:
                self._remove_all_custom_sections()
                self.custom_section_added = False
            except Exception:
                self.recorder.add_text("custom section", "remove skipped")
                return
        snapshot = self._generate_pdf_snapshot("pdf_no_custom")
        self._assert_pdf_missing(
            snapshot,
            [self.custom_section_title, self.custom_section_bullets[0]],
            "custom section",
        )
        self._assert_pdf_contains(
            snapshot, [self.section_title_overrides["Experience"]], "custom section"
        )

    def _remove_all_custom_sections(self) -> None:
        panel = self.ui.left_panel()
        deadline = time.time() + (self.cfg.timeout_ms / 1000.0)
        while time.time() < deadline:
            cards = panel.locator('[data-custom-section="1"]')
            count = cards.count()
            if count == 0:
                return
            card = cards.first
            btn = card.get_by_role("button", name="Remove")
            if btn.count() == 0:
                return
            try:
                btn.first.scroll_into_view_if_needed(timeout=self.cfg.timeout_ms)
                btn.first.click(timeout=self.cfg.action_timeout_ms)
            except Exception:
                return
            wait_deadline = time.time() + 5.0
            while time.time() < wait_deadline:
                if cards.count() < count:
                    break
                time.sleep(0.1)
            time.sleep(0.2)
        return

    def _pdf_auto_fit_one_page(self) -> None:
        self.ui.set_switch("Auto-fit", True)
        self.ui.set_auto_fit_pages("1")
        self.ui.set_switch("Include Skills section", False)
        self._set_section_visibility_by_title(
            self.section_title_overrides["Experience"], False
        )
        self._set_section_visibility_by_title(
            self.section_title_overrides["Education"], False
        )
        self._set_section_visibility_by_title(
            self.section_title_overrides["Startup Founder"], False
        )
        snapshot = self._generate_pdf_snapshot("pdf_auto_fit_one_page")
        self._assert_pdf_pages(snapshot, at_most=2)
        self._assert_pdf_contains(snapshot, [self.summary_head], "auto-fit")

    def _generate_pdf(self) -> None:
        self.ui.fill_placeholder("Paste or type the job requisition here", "")
        self.ui.click_button("Generate PDF")
        self.ui.wait_for_pdf()


def _parse_args() -> UIConfig:
    parser = argparse.ArgumentParser(
        description="Playwright end-to-end UI checks for the Resume Builder."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("PLAYWRIGHT_URL", "http://localhost:3000"),
        help="URL for the running app.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Timeout in seconds.",
    )
    parser.add_argument(
        "--pdf-timeout",
        type=float,
        default=45.0,
        help="PDF embed timeout in seconds.",
    )
    parser.add_argument(
        "--allow-llm-error",
        action="store_true",
        help="Allow LLM errors without failing the run.",
    )
    parser.add_argument(
        "--allow-db-error",
        action="store_true",
        help="Allow DB errors without failing the run.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run with a visible browser window.",
    )
    parser.add_argument(
        "--slowmo",
        type=int,
        default=0,
        help="Slow down Playwright actions in ms.",
    )
    parser.add_argument(
        "--screenshot-dir",
        default="",
        help="Directory for screenshots on failure.",
    )
    args = parser.parse_args()

    target = (args.url or "").strip()
    if not target:
        raise SystemExit("Missing --url.")
    timeout_ms = max(1000, int(args.timeout * 1000))
    pdf_timeout_ms = max(2000, int(args.pdf_timeout * 1000))
    parsed = urlparse(target)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else target

    return UIConfig(
        base_url=origin,
        timeout_ms=timeout_ms,
        action_timeout_ms=min(5000, max(800, int(timeout_ms * 0.25))),
        pdf_timeout_ms=pdf_timeout_ms,
        allow_llm_error=bool(args.allow_llm_error),
        allow_db_error=bool(args.allow_db_error),
        headed=bool(args.headed),
        slowmo_ms=int(args.slowmo or 0),
        screenshot_dir=(args.screenshot_dir or None),
    )


def _attach_console_listeners(page, recorder: IssueRecorder) -> None:
    def _on_console(msg) -> None:
        try:
            if msg.type == "error":
                recorder.add_text("console error", msg.text)
        except Exception:
            return

    def _on_page_error(err) -> None:
        recorder.add_text("page error", str(err))

    page.on("console", _on_console)
    page.on("pageerror", _on_page_error)


def main() -> int:
    cfg = _parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"Playwright not available: {exc}")
        return 2

    recorder = IssueRecorder()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not cfg.headed, slow_mo=cfg.slowmo_ms)
        context = browser.new_context()
        try:
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"], origin=cfg.base_url
            )
        except Exception:
            pass
        page = context.new_page()
        page.set_default_timeout(cfg.timeout_ms)
        _attach_console_listeners(page, recorder)
        ui = ResumeUI(page, recorder, cfg)
        runner = UITestRunner(ui, recorder, cfg)
        runner.run()
        try:
            recorder.raise_if_any()
        finally:
            context.close()
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
