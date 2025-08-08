from __future__ import annotations

import random
from typing import List

try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover - optional
    nltk = None
    wn = None

from .data_models import Document


def maybe_prepare_nltk() -> None:
    if not nltk:
        return
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        # best-effort local download; will fail offline
        try:
            nltk.download("wordnet", quiet=True)
        except Exception:
            pass


def synonym_replacement(doc: Document, p: float = 0.05) -> Document:
    if not wn:
        return doc
    words = doc.text.split()
    new_words: list[str] = []
    for w in words:
        if random.random() < p:
            syns = wn.synsets(w)
            lemmas = {l.name().replace("_", " ") for s in syns for l in s.lemmas()}
            lemmas.discard(w)
            if lemmas:
                new_words.append(random.choice(sorted(list(lemmas))))
                continue
        new_words.append(w)
    return Document(text=" ".join(new_words), language=doc.language, metadata=doc.metadata)


def sentence_reorder(doc: Document, p: float = 0.2) -> Document:
    sentences = doc.text.split(". ")
    if random.random() < p and len(sentences) > 2:
        random.shuffle(sentences)
    return Document(text=". ".join(sentences), language=doc.language, metadata=doc.metadata)


def augment_documents(documents: List[Document]) -> List[Document]:
    maybe_prepare_nltk()
    out: list[Document] = []
    for d in documents:
        d1 = synonym_replacement(d)
        d2 = sentence_reorder(d1)
        out.append(d2)
    return out
