from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import sentencepiece as spm

from .data_models import Document, TokenizedChunk


def train_or_load_sp_model(
    texts: Iterable[str],
    model_dir: Path,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    model_prefix: str = "spm",
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_prefix}.model"
    if model_path.exists():
        return model_path

    input_path = model_dir / "_train_input.txt"
    with open(input_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(model_dir / model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
    )
    return model_path


def tokenize_and_chunk(
    documents: List[Document],
    model_path: Path,
    chunk_size: int,
) -> List[TokenizedChunk]:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    chunks: list[TokenizedChunk] = []
    for d in documents:
        ids = sp.encode(d.text, out_type=int)
        for i in range(0, len(ids), chunk_size):
            window = ids[i : i + chunk_size]
            text_piece = sp.decode(window)
            chunks.append(TokenizedChunk(tokens=list(window), text=text_piece))
    return chunks
