from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


APP_NAME = "CorpusForgeMk1"


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    data_dir: Path
    cache_dir: Path
    logs_dir: Path
    db_path: Path
    licenses_dir: Path


@dataclass
class AppConfig:
    fasttext_model_path: Optional[Path] = None
    sentencepiece_model_dir: Optional[Path] = None
    dark_mode: bool = True

    @staticmethod
    def config_file(paths: AppPaths) -> Path:
        return paths.data_dir / "config.json"

    @classmethod
    def load(cls, paths: AppPaths) -> "AppConfig":
        cf = cls.config_file(paths)
        if cf.exists():
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                fasttext_model_path=Path(data["fasttext_model_path"]) if data.get("fasttext_model_path") else None,
                sentencepiece_model_dir=Path(data["sentencepiece_model_dir"]) if data.get("sentencepiece_model_dir") else None,
                dark_mode=bool(data.get("dark_mode", True)),
            )
        return cls()

    def save(self, paths: AppPaths) -> None:
        cf = self.config_file(paths)
        cf.parent.mkdir(parents=True, exist_ok=True)
        with open(cf, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fasttext_model_path": str(self.fasttext_model_path) if self.fasttext_model_path else None,
                    "sentencepiece_model_dir": str(self.sentencepiece_model_dir) if self.sentencepiece_model_dir else None,
                    "dark_mode": self.dark_mode,
                },
                f,
                indent=2,
            )


def discover_base_dir() -> Path:
    # Prefer E: drive when present (Windows), otherwise user home
    if platform.system() == "Windows":
        e_drive = Path("E:/")
        if e_drive.exists():
            return e_drive / APP_NAME
    return Path.home() / f".{APP_NAME}"


def get_paths() -> AppPaths:
    base = discover_base_dir()
    data_dir = base / "data"
    cache_dir = base / "cache"
    logs_dir = base / "logs"
    licenses_dir = base / "licenses"
    db_path = data_dir / "metadata.sqlite3"

    for p in [data_dir, cache_dir, logs_dir, licenses_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return AppPaths(
        base_dir=base,
        data_dir=data_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        db_path=db_path,
        licenses_dir=licenses_dir,
    )


def configure_environment() -> None:
    paths = get_paths()
    # Honor E: drive preference for Hugging Face caches [[memory:4630939]] [[memory:3675063]]
    os.environ.setdefault("HF_HOME", str(paths.cache_dir / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths.cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths.cache_dir / "transformers"))
    os.environ.setdefault("SENTENCEPIECE_HOME", str(paths.cache_dir / "sentencepiece"))

    (paths.cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
    (paths.cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (paths.cache_dir / "transformers").mkdir(parents=True, exist_ok=True)
    (paths.cache_dir / "sentencepiece").mkdir(parents=True, exist_ok=True)
