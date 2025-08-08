# CorpusForge Mk1

Offline AI Dataset Factory for building, cleaning, tokenizing, and exporting datasets.

## Features
- Import: .txt, .csv, .jsonl, .parquet, .zip
- Cleaning: dedup (MinHash), language filter (FastText), profanity/NSFW filters, regex cleaning, unicode normalization
- Tokenization & chunking: SentencePiece, configurable chunk sizes
- Augmentation: synonym replacement (optional), sentence reordering, dataset merging/shuffling, balancing
- Export: .jsonl, .txt, .parquet, .csv + metadata.json with stats and dataset hash
- Piracy protection: offline license check (RSA public key), AES-encrypted license storage, invisible watermark in exports
- UI: Flet desktop with sidebar pipeline steps, logs, progress, dark mode
- Performance: multithreaded, fully offline

## Install
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## FastText language identification
Download a FastText lid.176.bin model and set its path in Settings or pass via CLI.

## Run
- UI
```bash
python main.py --ui
```
- CLI
```bash
python main.py \
  --inputs path/to/folder path/to/file.txt \
  --output-dir out_dir \
  --fasttext-model E:/models/lid.176.bin \
  --chunk-size 1024 \
  --export jsonl parquet
```

## Packaging (Windows example)
```bash
pyinstaller --noconfirm --onefile --name corpusforge --add-data "assets/*;assets" main.py
```

## Notes
- Hugging Face cache paths default to E: drive when available.
- License is validated offline; see `utils/license.py`.
