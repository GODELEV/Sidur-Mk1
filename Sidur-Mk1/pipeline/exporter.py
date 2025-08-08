from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from .data_models import DatasetStats, Document, TokenizedChunk


ZERO_WIDTH_SPACE = "\u200b"
ZERO_WIDTH_JOINER = "\u200d"


def _compute_dataset_hash(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _watermark_text(text: str, watermark_hex: str) -> str:
    # Encode hex as invisible characters
    bitstream = "".join(f"{int(ch, 16):04b}" for ch in watermark_hex)
    water = []
    for i, ch in enumerate(text):
        if i < len(bitstream):
            marker = ZERO_WIDTH_SPACE if bitstream[i] == "0" else ZERO_WIDTH_JOINER
            water.append(ch + marker)
        else:
            water.append(ch)
    return "".join(water)


def export_all(
    output_dir: Path,
    documents: List[Document],
    chunks: List[TokenizedChunk],
    formats: List[str],
) -> DatasetStats:
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [d.text for d in documents]
    dataset_hash = _compute_dataset_hash(texts)
    languages: dict[str, int] = {}
    for d in documents:
        if d.language:
            languages[d.language] = languages.get(d.language, 0) + 1

    # Exports
    if "jsonl" in formats:
        with open(output_dir / "dataset.jsonl", "w", encoding="utf-8") as f:
            for d in documents:
                wt = _watermark_text(d.text, dataset_hash[:16])
                f.write(json.dumps({"text": wt, "language": d.language}) + "\n")

    if "txt" in formats:
        with open(output_dir / "dataset.txt", "w", encoding="utf-8") as f:
            for d in documents:
                f.write(_watermark_text(d.text, dataset_hash[:16]) + "\n")

    if "csv" in formats:
        df = pd.DataFrame({"text": texts, "language": [d.language for d in documents]})
        df.to_csv(output_dir / "dataset.csv", index=False)

    if "parquet" in formats:
        df = pd.DataFrame({"text": texts, "language": [d.language for d in documents]})
        df.to_parquet(output_dir / "dataset.parquet", index=False)

    # Token dumps (optional)
    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"tokens": c.tokens, "text": c.text}) + "\n")

    stats = DatasetStats(
        num_documents=len(documents),
        num_tokens=sum(len(c.tokens) for c in chunks),
        language_distribution=languages,
        dataset_hash=dataset_hash,
    )
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_documents": stats.num_documents,
                "num_tokens": stats.num_tokens,
                "language_distribution": stats.language_distribution,
                "dataset_hash": stats.dataset_hash,
            },
            f,
            indent=2,
        )
    return stats
