from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import flet as ft

from utils.config import AppConfig, configure_environment, get_paths
from utils.license import LicenseManager
from utils.logger import get_logger
from pipeline.manager import PipelineConfig, PipelineManager


LOGGER = get_logger(__name__)


def launch_app() -> None:
    ft.app(target=_main)


def _main(page: ft.Page) -> None:
    configure_environment()
    page.title = "CorpusForge Mk1"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1280
    page.window_height = 840
    page.padding = 10

    paths = get_paths()
    cfg = AppConfig.load(paths)

    # License (best effort)
    lic = LicenseManager().load_and_validate(password="local")
    if not lic:
        page.snack_bar = ft.SnackBar(ft.Text("No license found. Some features may be restricted."))
        page.snack_bar.open = True

    # ---------- State ----------
    selected_inputs: list[Path] = []
    active_section: str = "import"
    running_task: Optional[PipelineManager] = None

    # Common controls used across sections
    progress_bar = ft.ProgressBar(width=300)
    progress_text = ft.Text("Idle")
    log_view = ft.Text(value="Welcome to CorpusForge Mk1", selectable=True, max_lines=4)
    inspector_list = ft.ListView(expand=True, height=240, spacing=6)
    lang_breakdown = ft.Column(spacing=6)
    raw_count_text = ft.Text("0")
    clean_count_text = ft.Text("0")
    chunk_count_text = ft.Text("0")

    # Inputs panel controls
    output_dir_field = ft.TextField(label="Output directory", value=str(paths.base_dir / "exports"), width=460)
    fasttext_field = ft.TextField(label="FastText model path (optional)", value=str(cfg.fasttext_model_path or ""), width=460)

    # Clean panel controls
    regex_chk = ft.Checkbox(label="Regex clean (emails/URLs/HTML)", value=True)
    profanity_chk = ft.Checkbox(label="Profanity filter", value=True)
    lang_filter_chk = ft.Checkbox(label="Language filter (FastText)", value=True)
    dedup_chk = ft.Checkbox(label="Remove duplicates (MinHash)", value=True)
    lang_whitelist_field = ft.TextField(label="Language whitelist (e.g. en,es)")
    lang_blacklist_field = ft.TextField(label="Language blacklist (e.g. ru,zh)")

    # Tokenize panel controls
    sp_model_field = ft.TextField(label="SentencePiece model file or directory")
    sp_vocab_field = ft.TextField(label="SPM vocab size if training", value="32000", width=200)
    chunk_field = ft.TextField(label="Chunk size", value="1024", width=200)
    augment_chk = ft.Checkbox(label="Enable light augmentation", value=False)

    # Export panel controls
    jsonl_chk = ft.Checkbox(label="JSONL", value=True)
    txt_chk = ft.Checkbox(label="TXT", value=False)
    parquet_chk = ft.Checkbox(label="Parquet", value=False)
    csv_chk = ft.Checkbox(label="CSV", value=False)
    watermark_chk = ft.Checkbox(label="Watermark text (invisible)", value=False)
    ascii_safe_chk = ft.Checkbox(label="ASCII-safe JSON (escape non-ASCII)", value=False)

    # ---------- Helpers ----------
    def palette() -> dict:
        dark = page.theme_mode == ft.ThemeMode.DARK
        return {
            "card": "#191919" if dark else "#f3f3f3",
            "box": "#171717" if dark else "#ffffff",
            "muted": "#9aa0a6" if dark else "#5f6368",
            "accent": "#3b82f6",
        }

    def card(title: str, body_controls: list[ft.Control], expand: bool = False, width: Optional[int] = None) -> ft.Container:
        return ft.Container(
            bgcolor=palette()["card"],
            padding=12,
            border_radius=10,
            border=ft.border.all(1, "#2a2a2a"),
            content=ft.Column([ft.Text(title, weight=ft.FontWeight.BOLD)] + body_controls, spacing=8),
            expand=expand,
            width=width,
        )

    def log(msg: str) -> None:
        LOGGER.info(msg)
        log_view.value = msg
        page.update()

    # Sidebar actions
    def set_section(name: str) -> None:
        nonlocal active_section
        active_section = name
        main_panel.content = build_section()
        page.update()

    # File picker
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            for f in e.files:
                selected_inputs.append(Path(f.path))
            refresh_input_list()
            log(f"Selected {len(e.files)} inputs")

    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    input_list = ft.ListView(height=160, expand=False)
    def refresh_input_list() -> None:
        input_list.controls = [ft.Text(str(p)) for p in selected_inputs]

    def run_clicked(e: ft.ControlEvent) -> None:
        if not selected_inputs:
            log("No inputs selected")
            return
        out_dir = Path(output_dir_field.value or (paths.base_dir / "exports"))
        out_dir.mkdir(parents=True, exist_ok=True)

        config = PipelineConfig(
            input_paths=selected_inputs.copy(),
            output_dir=out_dir,
            fasttext_model_path=Path(fasttext_field.value) if fasttext_field.value else None,
            chunk_size=int(chunk_field.value or 1024),
            export_formats=[fmt for fmt, chk in [("jsonl", jsonl_chk), ("txt", txt_chk), ("parquet", parquet_chk), ("csv", csv_chk)] if chk.value],
            language_whitelist=set(filter(None, (lang_whitelist_field.value or "").split(","))),
            language_blacklist=set(filter(None, (lang_blacklist_field.value or "").split(","))),
            enable_augmentation=augment_chk.value,
            sentencepiece_model_path=Path(sp_model_field.value) if sp_model_field.value else None,
            sentencepiece_vocab_size=int(sp_vocab_field.value or 32000),
            num_workers=4,
            enable_regex_clean=regex_chk.value,
            enable_profanity_filter=profanity_chk.value,
            enable_language_filter=lang_filter_chk.value,
            enable_deduplication=dedup_chk.value,
            on_progress=on_progress_cb,
            on_preview=on_preview_cb,
            on_metrics=on_metrics_cb,
            watermark_texts=watermark_chk.value,
            ascii_safe_json=ascii_safe_chk.value,
        )

        log("Running pipeline...")
        progress_bar.value = None
        page.update()

        async def _run():
            nonlocal running_task
            loop = asyncio.get_running_loop()

            def blocking() -> bool:
                nonlocal running_task
                mgr = PipelineManager(config)
                running_task = mgr
                ok_local = mgr.run()
                running_task = None
                return ok_local

            ok = await loop.run_in_executor(None, blocking)
            progress_bar.value = 0
            log("Done" if ok else "Failed")
            page.update()

        page.run_task(_run)

    # Callbacks from pipeline
    def on_progress_cb(stage: str, value: float, message: str) -> None:
        def _ui():
            progress_text.value = f"{stage}: {message}"
            progress_bar.value = value if value is not None else None
            page.update()
        page.call_from_thread(_ui)

    def on_preview_cb(stage: str, sample: List[str]) -> None:
        def _ui():
            inspector_list.controls = [ft.Text(s[:400]) for s in sample]
            page.update()
        page.call_from_thread(_ui)

    def on_metrics_cb(stage: str, payload: dict) -> None:
        def _ui():
            if stage == "clean" and payload.get("language_distribution"):
                lang_breakdown.controls.clear()
                dist = payload["language_distribution"]
                total = sum(dist.values()) or 1
                for code, count in sorted(dist.items(), key=lambda x: -x[1])[:8]:
                    pct = int(count * 100 / total)
                    lang_breakdown.controls.append(
                        ft.Row([ft.Text(code.upper(), width=60), ft.ProgressBar(value=pct / 100, width=180), ft.Text(f"{pct}%")])
                    )
                clean_count_text.value = str(payload.get("clean_count", 0))
            if stage == "import":
                raw_count_text.value = str(payload.get("raw_count", 0))
            if stage == "tokenize":
                chunk_count_text.value = str(payload.get("chunk_count", 0))
            page.update()
        page.call_from_thread(_ui)

    # ---------- Build Sections ----------
    def build_import_section() -> ft.Control:
        add_btn = ft.ElevatedButton(text="Add files", on_click=lambda e: file_picker.pick_files(allow_multiple=True))
        return card(
            "Import",
            [
                ft.Row([add_btn]),
                ft.Container(content=input_list, bgcolor=palette()["card"], border_radius=8, padding=8),
                output_dir_field,
                fasttext_field,
            ],
            expand=True,
        )

    def build_clean_section() -> ft.Control:
        return card(
            "Cleaning",
            [
                regex_chk,
                profanity_chk,
                lang_filter_chk,
                dedup_chk,
                lang_whitelist_field,
                lang_blacklist_field,
            ],
            expand=True,
        )

    def build_tokenize_section() -> ft.Control:
        return card(
            "Tokenization",
            [
                sp_model_field,
                ft.Row([sp_vocab_field, chunk_field]),
                augment_chk,
            ],
            expand=True,
        )

    def build_export_section() -> ft.Control:
        return card(
            "Export",
            [
                ft.Row([jsonl_chk, txt_chk, parquet_chk, csv_chk], wrap=True),
                ft.Row([watermark_chk, ascii_safe_chk], wrap=True),
            ],
            expand=True,
        )

    def build_section() -> ft.Control:
        if active_section == "import":
            return build_import_section()
        if active_section == "clean":
            return build_clean_section()
        if active_section == "tokenize":
            return build_tokenize_section()
        return build_export_section()

    # Sidebar
    sidebar = ft.Container(
        width=260,
        content=ft.Column(
            [
                ft.Text("Sections", weight=ft.FontWeight.BOLD),
                ft.TextButton("Import", on_click=lambda e: set_section("import")),
                ft.TextButton("Clean", on_click=lambda e: set_section("clean")),
                ft.TextButton("Tokenize", on_click=lambda e: set_section("tokenize")),
                ft.TextButton("Export", on_click=lambda e: set_section("export")),
                ft.Divider(),
                ft.Text("Summary", weight=ft.FontWeight.BOLD),
                ft.Row([ft.Text("Raw:"), raw_count_text]),
                ft.Row([ft.Text("Cleaned:"), clean_count_text]),
                ft.Row([ft.Text("Chunks:"), chunk_count_text]),
                ft.Divider(),
                ft.Text("Pipeline", weight=ft.FontWeight.BOLD),
                progress_bar,
                progress_text,
            ],
            spacing=8,
        ),
    )

    # Main panel switches between sections
    main_panel = ft.Container(expand=True, content=build_section())

    # Right side: inspector and language
    right_panel = ft.Column(
        [
            card("Inspector", [inspector_list], expand=False),
            card("Language breakdown", [lang_breakdown], expand=True),
        ],
        expand=False,
        width=360,
        spacing=10,
    )

    # Footer controls
    run_btn = ft.ElevatedButton(text="Start", on_click=run_clicked)
    cancel_btn = ft.OutlinedButton(text="Cancel", on_click=lambda e: running_task.cancel() if running_task else None)
    footer = ft.Row([run_btn, cancel_btn, ft.Container(expand=True), ft.Text("Dark"), ft.Switch(value=True, on_change=lambda e: toggle_theme())])

    def toggle_theme() -> None:
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        page.update()

    # Header
    header = ft.Row([ft.Text("CorpusForge Mk1", size=20, weight=ft.FontWeight.BOLD), ft.Container(expand=True)])

    # Layout assembly
    page.add(
        ft.Column(
            [
                header,
                ft.Row([sidebar, main_panel, right_panel], expand=True),
                card("Status / Logs", [log_view], expand=False),
                footer,
            ],
            expand=True,
            spacing=10,
        )
    )

    refresh_input_list()
    page.update()

