from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from utils.logger import get_logger
from utils.db import insert_dataset
from .augmentation import augment_documents
from .cleaning import clean_documents
from .data_models import Document, TokenizedChunk
from .exporter import export_all
from .input_importer import import_documents
from .tokenization import tokenize_and_chunk, train_or_load_sp_model


LOGGER = get_logger(__name__)


@dataclass
class PipelineConfig:
    input_paths: List[Path]
    output_dir: Path
    fasttext_model_path: Optional[Path]
    chunk_size: int
    export_formats: List[str]
    language_whitelist: set[str]
    language_blacklist: set[str]
    enable_augmentation: bool
    sentencepiece_model_path: Optional[Path]
    sentencepiece_vocab_size: int
    num_workers: int
    # Cleaning toggles
    enable_regex_clean: bool = True
    enable_profanity_filter: bool = True
    enable_language_filter: bool = True
    enable_deduplication: bool = True

    # Callback hooks
    on_progress: Optional[callable] = None  # (stage: str, value: float, message: str) -> None
    on_preview: Optional[callable] = None   # (stage: str, sample: list[str]) -> None


class PipelineManager:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _check_cancel(self) -> bool:
        if self.cancel_event.is_set():
            LOGGER.warning("Pipeline cancelled")
            return True
        return False

    def _emit_progress(self, stage: str, value: float, message: str) -> None:
        if self.config.on_progress:
            try:
                self.config.on_progress(stage, value, message)
            except Exception:
                pass

    def _emit_preview(self, stage: str, sample: List[str]) -> None:
        if self.config.on_preview:
            try:
                self.config.on_preview(stage, sample)
            except Exception:
                pass

    def run(self) -> bool:
        LOGGER.info("Starting pipeline")
        if self._check_cancel():
            return False

        # Import
        LOGGER.info("Importing inputs...")
        self._emit_progress("import", 0.0, "Starting import")
        docs = import_documents(self.config.input_paths)
        self._emit_progress("import", 1.0, f"Imported {len(docs)} items")
        self._emit_preview("import", [d.text[:400] for d in docs[:5]])
        LOGGER.info("Imported %d raw items", len(docs))
        if self._check_cancel():
            return False
        if not docs:
            LOGGER.error("No documents imported")
            return False

        # Cleaning
        LOGGER.info("Cleaning documents...")
        self._emit_progress("clean", 0.0, "Starting cleaning")
        docs = clean_documents(
            docs,
            language_whitelist=self.config.language_whitelist or None,
            language_blacklist=self.config.language_blacklist or None,
            fasttext_model_path=str(self.config.fasttext_model_path) if self.config.fasttext_model_path else None,
            enable_regex_clean=self.config.enable_regex_clean,
            enable_profanity_filter=self.config.enable_profanity_filter,
            enable_language_filter=self.config.enable_language_filter,
            enable_deduplication=self.config.enable_deduplication,
        )
        self._emit_progress("clean", 1.0, f"Cleaned: {len(docs)} remain")
        self._emit_preview("clean", [d.text[:400] for d in docs[:5]])
        LOGGER.info("Kept %d documents after cleaning", len(docs))
        if self._check_cancel():
            return False

        # Augmentation (optional)
        if self.config.enable_augmentation:
            LOGGER.info("Augmenting documents...")
            self._emit_progress("augment", 0.0, "Starting augmentation")
            docs = augment_documents(docs)
            self._emit_progress("augment", 1.0, f"Augmentation produced {len(docs)} docs")
            LOGGER.info("Documents after augmentation: %d", len(docs))
            if self._check_cancel():
                return False

        # Tokenization & chunking
        LOGGER.info("Tokenizing & chunking...")
        self._emit_progress("tokenize", 0.0, "Preparing tokenizer")
        if self.config.sentencepiece_model_path and self.config.sentencepiece_model_path.is_file():
            sp_model = self.config.sentencepiece_model_path
        else:
            model_dir = self.config.sentencepiece_model_path or (self.config.output_dir / "spm")
            sp_model = train_or_load_sp_model((d.text for d in docs), model_dir)
        chunks = tokenize_and_chunk(docs, sp_model, self.config.chunk_size)
        self._emit_progress("tokenize", 1.0, f"Chunks: {len(chunks)}")
        LOGGER.info("Created %d chunks", len(chunks))
        if self._check_cancel():
            return False

        # Export
        LOGGER.info("Exporting...")
        self._emit_progress("export", 0.0, "Starting export")
        stats = export_all(self.config.output_dir, docs, chunks, self.config.export_formats)
        self._emit_progress("export", 1.0, "Export finished")
        LOGGER.info("Export complete: %s", self.config.output_dir)

        # Record metadata
        insert_dataset(
            name=self.config.output_dir.name,
            num_documents=stats.num_documents,
            num_tokens=stats.num_tokens,
            languages=stats.language_distribution.keys(),
            dataset_hash=stats.dataset_hash,
            output_dir=self.config.output_dir,
        )

        LOGGER.info("Pipeline finished successfully")
        return True
