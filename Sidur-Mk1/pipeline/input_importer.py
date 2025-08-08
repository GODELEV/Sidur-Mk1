from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from .data_models import Document


TEXT_EXTS = {".txt"}
CSV_EXTS = {".csv"}
JSONL_EXTS = {".jsonl"}
PARQUET_EXTS = {".parquet"}
ZIP_EXTS = {".zip"}


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    yield child
        elif p.is_file():
            yield p


def import_documents(paths: Iterable[Path]) -> List[Document]:
    docs: list[Document] = []
    for file_path in iter_files(paths):
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTS:
            docs.extend(_import_txt(file_path))
        elif suffix in CSV_EXTS:
            docs.extend(_import_csv(file_path))
        elif suffix in JSONL_EXTS:
            docs.extend(_import_jsonl(file_path))
        elif suffix in PARQUET_EXTS:
            docs.extend(_import_parquet(file_path))
        elif suffix in ZIP_EXTS:
            docs.extend(_import_zip(file_path))
    return docs


def _import_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [Document(text=line) for line in lines]


def _import_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path)
    # try common text columns
    for col in ("text", "content", "body"):
        if col in df.columns:
            return [Document(text=str(x)) for x in df[col].dropna().astype(str).tolist()]
    # fallback: concatenate row cells
    return [Document(text=" ".join(map(str, row.values()))) for _, row in df.iterrows()]


def _import_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = obj.get("text") or obj.get("content") or obj.get("body")
                    if text:
                        docs.append(Document(text=str(text), metadata=obj))
                else:
                    docs.append(Document(text=str(obj)))
            except json.JSONDecodeError:
                docs.append(Document(text=line))
    return docs


def _import_parquet(path: Path) -> list[Document]:
    df = pd.read_parquet(path)
    for col in ("text", "content", "body"):
        if col in df.columns:
            return [Document(text=str(x)) for x in df[col].dropna().astype(str).tolist()]
    return [Document(text=" ".join(map(str, row.values()))) for _, row in df.iterrows()]


def _import_zip(path: Path) -> list[Document]:
    docs: list[Document] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            with zf.open(name, "r") as f:
                data = f.read()
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if suffix in TEXT_EXTS:
                docs.extend([Document(text=line.strip()) for line in text.splitlines() if line.strip()])
            elif suffix in JSONL_EXTS:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            t = obj.get("text") or obj.get("content") or obj.get("body")
                            if t:
                                docs.append(Document(text=str(t), metadata=obj))
                        else:
                            docs.append(Document(text=str(obj)))
                    except json.JSONDecodeError:
                        docs.append(Document(text=line))
            elif suffix in CSV_EXTS:
                sio = io.StringIO(text)
                df = pd.read_csv(sio)
                for col in ("text", "content", "body"):
                    if col in df.columns:
                        docs.extend([Document(text=str(x)) for x in df[col].dropna().astype(str).tolist()])
                        break
    return docs
