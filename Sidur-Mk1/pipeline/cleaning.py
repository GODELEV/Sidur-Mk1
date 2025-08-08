from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List, Optional, Set, Tuple

try:
    import fasttext  # type: ignore
except Exception:  # pragma: no cover - optional
    fasttext = None

from datasketch import MinHash, MinHashLSH

from .data_models import Document


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Tiny profanity list placeholder; extend as needed
PROFANE = {
    "damn",
    "hell",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    return text.strip()


def regex_clean(text: str) -> str:
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_profanity(text: str) -> Optional[str]:
    tokens = re.findall(r"\w+", text.lower())
    if any(tok in PROFANE for tok in tokens):
        return None
    return text


def detect_language(texts: Iterable[str], model_path: Optional[str]) -> List[Optional[str]]:
    if not model_path or not fasttext:
        return [None for _ in texts]
    model = fasttext.load_model(model_path)
    langs: list[Optional[str]] = []
    for t in texts:
        if not t:
            langs.append(None)
            continue
        label, prob = model.predict(t.replace("\n", " ")[:1000])  # use prefix
        code = label[0].replace("__label__", "") if label else None
        langs.append(code)
    return langs


def deduplicate_documents(documents: List[Document], num_perm: int = 128, threshold: float = 0.9) -> List[Document]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[Tuple[str, MinHash]] = []
    kept: list[Document] = []

    for idx, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        for token in set(doc.text.split()):
            mh.update(token.encode("utf-8", errors="ignore"))
        key = f"doc_{idx}"
        minhashes.append((key, mh))

    for key, mh in minhashes:
        dup = lsh.query(mh)
        if dup:
            # already similar to an inserted doc
            continue
        lsh.insert(key, mh)
        kept.append(documents[int(key.split("_")[-1])])

    return kept


def clean_documents(
    documents: List[Document],
    language_whitelist: Set[str] | None = None,
    language_blacklist: Set[str] | None = None,
    fasttext_model_path: Optional[str] = None,
    enable_regex_clean: bool = True,
    enable_profanity_filter: bool = True,
    enable_language_filter: bool = True,
    enable_deduplication: bool = True,
) -> List[Document]:
    normalized = [normalize_text(d.text) for d in documents]
    if enable_regex_clean:
        normalized = [regex_clean(t) for t in normalized]
    normalized_docs = [Document(text=t) for t in normalized if t]

    # profanity filter
    filtered_docs: list[Document] = []
    for d in normalized_docs:
        if enable_profanity_filter:
            rt = remove_profanity(d.text)
            if rt is None:
                continue
            filtered_docs.append(Document(text=rt))
        else:
            filtered_docs.append(d)

    # language detection
    if enable_language_filter:
        langs = detect_language([d.text for d in filtered_docs], fasttext_model_path)
    else:
        langs = [None for _ in filtered_docs]

    lang_docs: list[Document] = []
    for d, lang in zip(filtered_docs, langs):
        if lang:
            if language_whitelist and lang not in language_whitelist:
                continue
            if language_blacklist and lang in language_blacklist:
                continue
        lang_docs.append(Document(text=d.text, language=lang))

    # deduplicate
    if enable_deduplication:
        deduped = deduplicate_documents(lang_docs)
    else:
        deduped = lang_docs

    return deduped
