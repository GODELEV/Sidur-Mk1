### CorpusForge Mk1 Architecture

```mermaid
graph TD;
  subgraph UI[Flet Desktop UI]
    A[Tabs: Import/Clean/Tokenize/Export]
    B[Controls: Start/Cancel, Progress, Inspector, Breakdown]
  end

  subgraph Pipeline
    P0[PipelineManager]
    P1[Importer]
    P2[Cleaning\n(regex, profanity, lang, dedup)]
    P3[Augmentation]
    P4[SentencePiece\nTrain/Load & Chunk]
    P5[Exporter\nJSONL/TXT/CSV/Parquet + metadata + watermark]
  end

  subgraph Utils
    U0[Config & Paths\n(HF caches on E:)]
    U1[Logger]
    U2[License\nAES + RSA offline]
    U3[SQLite Metadata]
  end

  A -->|User actions| P0
  B -->|Callbacks| P0
  P0 --> P1 --> P2 --> P3 --> P4 --> P5
  P0 -->|writes| U3
  UI -->|reads| U0
  P0 -->|logs| U1
  UI -->|check| U2
```
