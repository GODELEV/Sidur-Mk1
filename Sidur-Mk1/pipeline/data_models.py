from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    text: str
    language: Optional[str] = None  # ISO code
    metadata: Optional[dict] = None


@dataclass
class TokenizedChunk:
    tokens: list[int]
    text: str


@dataclass
class DatasetStats:
    num_documents: int
    num_tokens: int
    language_distribution: dict[str, int]
    dataset_hash: str
