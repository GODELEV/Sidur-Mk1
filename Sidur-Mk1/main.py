import argparse
import os
import sys
from pathlib import Path

from utils.config import AppConfig, configure_environment
from utils.license import LicenseManager
from utils.logger import get_logger
from pipeline.manager import PipelineManager, PipelineConfig


LOGGER = get_logger(__name__)


def run_cli(args: argparse.Namespace) -> int:
    configure_environment()

    # Offline license check (best effort, does not block development)
    lic_mgr = LicenseManager()
    lic = lic_mgr.load_and_validate(password="local")
    if not lic:
        LOGGER.warning("No valid license found. Some features may be restricted.")
    else:
        LOGGER.info("License: %s (%s)", lic.name, lic.email)

    config = PipelineConfig(
        input_paths=[Path(p) for p in args.inputs],
        output_dir=Path(args.output_dir),
        fasttext_model_path=Path(args.fasttext_model) if args.fasttext_model else None,
        chunk_size=args.chunk_size,
        export_formats=args.export,
        language_whitelist=set(args.lang_whitelist or []),
        language_blacklist=set(args.lang_blacklist or []),
        enable_augmentation=args.augment,
        sentencepiece_model_path=Path(args.sp_model) if args.sp_model else None,
        sentencepiece_vocab_size=args.sp_vocab_size,
        num_workers=args.workers,
    )

    manager = PipelineManager(config)
    success = manager.run()
    return 0 if success else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CorpusForge Mk1 - Offline AI Dataset Factory")
    parser.add_argument("--ui", action="store_true", help="Launch Flet UI")

    # CLI arguments
    parser.add_argument("--inputs", nargs="*", default=[], help="Input files or folders")
    parser.add_argument("--output-dir", default="output", help="Directory to write exports")
    parser.add_argument("--fasttext-model", default=None, help="Path to FastText language id model (lid.176.bin)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Token chunk size")
    parser.add_argument("--export", nargs="*", default=["jsonl"], choices=["jsonl", "txt", "parquet", "csv"], help="Export formats")
    parser.add_argument("--lang-whitelist", nargs="*", default=None, help="Keep only these languages (2-letter codes)")
    parser.add_argument("--lang-blacklist", nargs="*", default=None, help="Remove these languages (2-letter codes)")
    parser.add_argument("--augment", action="store_true", help="Enable simple data augmentation")
    parser.add_argument("--sp-model", default=None, help="Existing SentencePiece model path or directory for training")
    parser.add_argument("--sp-vocab-size", type=int, default=32000, help="SentencePiece vocab size when training")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of workers")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.ui:
        # Lazy import flet UI to avoid dependency for CLI-only usage
        from ui.main_view import launch_app

        configure_environment()
        launch_app()
        return 0

    if not args.inputs:
        print("No inputs provided. Use --ui or pass --inputs.")
        return 2

    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
