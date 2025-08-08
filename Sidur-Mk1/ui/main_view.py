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
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 1280
    page.window_height = 840
    page.padding = 10

    paths = get_paths()
    cfg = AppConfig.load(paths)

    # License check and basic activation dialog
    lic_mgr = LicenseManager()
    lic = lic_mgr.load_and_validate(password="local")
    if not lic:
        dlg = ft.AlertDialog(title=ft.Text("License Required"), content=ft.Text("No valid license found. Some features may be restricted."))
        page.dialog = dlg
        dlg.open = True
        page.update()

    selected_inputs: list[Path] = []

    progress_bar = ft.ProgressBar(width=300)
    progress_text = ft.Text("Idle")
    log_view = ft.Text(value="Welcome to CorpusForge Mk1", selectable=True, max_lines=8)
    preview_area = ft.Text("", selectable=True, max_lines=14)
    lang_breakdown = ft.Column(spacing=4)

    def log(msg: str) -> None:
        LOGGER.info(msg)
        log_view.value = msg
        page.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            for f in e.files:
                selected_inputs.append(Path(f.path))
            log(f"Selected {len(e.files)} inputs")

    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    input_list = ft.Text(value="No inputs selected")

    def refresh_input_list() -> None:
        if selected_inputs:
            input_list.value = "\n".join(str(p) for p in selected_inputs)
        else:
            input_list.value = "No inputs selected"
        page.update()

    running_task: Optional[PipelineManager] = None

    def on_run_clicked(e: ft.ControlEvent) -> None:
        out_dir = Path(output_dir_field.value or (paths.base_dir / "exports")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not selected_inputs:
            log("No inputs selected")
            return

        config = PipelineConfig(
            input_paths=selected_inputs.copy(),
            output_dir=out_dir,
            fasttext_model_path=Path(fasttext_field.value) if fasttext_field.value else None,
            chunk_size=int(chunk_field.value or 1024),
            export_formats=[fmt for fmt, chk in [("jsonl", jsonl_chk), ("txt", txt_chk), ("parquet", parquet_chk), ("csv", csv_chk)] if chk.value],
            language_whitelist=set(filter(None, lang_whitelist_field.value.split(","))) if lang_whitelist_field.value else set(),
            language_blacklist=set(filter(None, lang_blacklist_field.value.split(","))) if lang_blacklist_field.value else set(),
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

    def on_cancel_clicked(e: ft.ControlEvent) -> None:
        nonlocal_running = locals()
        # Using closure variable running_task from outer scope
        pass

    def on_add_files(e: ft.ControlEvent) -> None:
        file_picker.pick_files(allow_multiple=True)

    # Controls
    output_dir_field = ft.TextField(label="Output directory", value=str(paths.base_dir / "exports"), width=500)
    fasttext_field = ft.TextField(label="FastText model path (optional)", value=str(cfg.fasttext_model_path or ""), width=500)
    sp_model_field = ft.TextField(label="SentencePiece model dir or file (optional)", value=str(cfg.sentencepiece_model_dir or ""), width=500)
    sp_vocab_field = ft.TextField(label="SPM vocab size (if training)", value="32000", width=200)
    chunk_field = ft.TextField(label="Chunk size", value="1024", width=200)
    lang_whitelist_field = ft.TextField(label="Language whitelist (comma codes)", width=500)
    lang_blacklist_field = ft.TextField(label="Language blacklist (comma codes)", width=500)

    jsonl_chk = ft.Checkbox(label="JSONL", value=True)
    txt_chk = ft.Checkbox(label="TXT", value=False)
    parquet_chk = ft.Checkbox(label="Parquet", value=False)
    csv_chk = ft.Checkbox(label="CSV", value=False)
    augment_chk = ft.Checkbox(label="Enable augmentation", value=False)
    regex_chk = ft.Checkbox(label="Regex clean", value=True)
    profanity_chk = ft.Checkbox(label="Profanity filter", value=True)
    lang_filter_chk = ft.Checkbox(label="Language filter", value=True)
    dedup_chk = ft.Checkbox(label="Remove duplicates", value=True)

    run_btn = ft.ElevatedButton(text="Start", on_click=on_run_clicked)
    cancel_btn = ft.OutlinedButton(text="Cancel", on_click=lambda e: running_task.cancel() if running_task else None)
    add_btn = ft.ElevatedButton(text="Add files", on_click=on_add_files)

    # Layout
    # Tabs inspired by dashboard UI
    import_tab = ft.Column([
        ft.Row([add_btn]),
        ft.Container(content=ft.Text("Selected files"), padding=5),
        ft.Container(content=input_list, bgcolor="#191919", padding=10, border_radius=6, height=180, expand=False),
    ], spacing=10)

    clean_tab = ft.Column([
        ft.Text("Cleaning Options", weight=ft.FontWeight.BOLD),
        ft.Row([regex_chk, profanity_chk]),
        ft.Row([lang_filter_chk, dedup_chk]),
        lang_whitelist_field,
        lang_blacklist_field,
    ], spacing=10)

    tokenize_tab = ft.Column([
        ft.Text("Tokenization", weight=ft.FontWeight.BOLD),
        sp_model_field,
        ft.Row([sp_vocab_field, chunk_field]),
        augment_chk,
    ], spacing=10)

    export_tab = ft.Column([
        ft.Text("Export Formats", weight=ft.FontWeight.BOLD),
        ft.Row([jsonl_chk, txt_chk, parquet_chk, csv_chk]),
        output_dir_field,
    ], spacing=10)

    control_row = ft.Row([run_btn, cancel_btn, progress_bar, progress_text], alignment=ft.MainAxisAlignment.START)
    preview_card = ft.Container(
        content=ft.Column([
            ft.Text("Inspector", weight=ft.FontWeight.BOLD),
            preview_area,
        ]),
        bgcolor="#191919",
        padding=10,
        border_radius=6,
        expand=True,
    )

    breakdown_card = ft.Container(
        content=ft.Column([
            ft.Text("Language breakdown", weight=ft.FontWeight.BOLD),
            lang_breakdown,
        ]),
        bgcolor="#191919",
        padding=10,
        border_radius=6,
        width=360,
    )

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Import", content=import_tab),
            ft.Tab(text="Clean", content=clean_tab),
            ft.Tab(text="Tokenize", content=tokenize_tab),
            ft.Tab(text="Export", content=export_tab),
        ],
        expand=1,
    )

    top_bar = ft.Row([
        ft.Text("CorpusForge Mk1", size=20, weight=ft.FontWeight.BOLD),
        ft.Container(expand=True),
        ft.Icon(name="nightlight_round"),
    ])

    page.add(
        ft.Column([
            top_bar,
            tabs,
            control_row,
            ft.Row([preview_card, breakdown_card], expand=True),
            ft.Container(content=ft.Column([ft.Text("Status / Logs", weight=ft.FontWeight.BOLD), log_view]), padding=0),
        ], expand=True)
    )

    # Drag-and-drop support
    # Drag & drop (fallbacks for older Flet)
    def _append_paths(paths: List[str]) -> None:
        for p in paths:
            if p:
                selected_inputs.append(Path(p))
        refresh_input_list()

    def on_drop_event(e) -> None:  # best-effort across versions
        paths: list[str] = []
        if getattr(e, "files", None):
            for f in e.files:
                if getattr(f, "path", None):
                    paths.append(f.path)
        _append_paths(paths)

    try:
        drop_target = ft.DragTarget(
            group="files",
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Drop files here", size=16),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                height=120,
                bgcolor="#171717",
                border_radius=8,
            ),
            on_drop=on_drop_event,  # new API
        )
    except TypeError:
        # Older API uses on_accept but won't deliver OS file paths; keep visual area and rely on FilePicker
        drop_target = ft.DragTarget(
            group="files",
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Drop files here (or use Add files)", size=16),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                height=120,
                bgcolor="#171717",
                border_radius=8,
            ),
        )
    import_tab.controls.append(drop_target)

    def on_progress_cb(stage: str, value: float, message: str) -> None:
        def _ui_update():
            progress_text.value = f"{stage}: {message}"
            if value is None:
                progress_bar.value = None
            else:
                progress_bar.value = value
            page.update()
        page.call_from_thread(_ui_update)

    def on_preview_cb(stage: str, sample: List[str]) -> None:
        def _ui_update():
            joined = "\n\n".join(sample)
            preview_area.value = joined
            page.update()
        page.call_from_thread(_ui_update)

    page.update()
