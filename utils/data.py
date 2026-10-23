"""Load the AraSeg TSV (or a subset JSON) into Document trees, and build word-level gold labels.

TSV columns used: new_split, doc_name, paragraph_id, sent_id_in_paragraph, text, source.
Each row is one gold sentence. A Document is a list of Paragraphs, each a list of Sentences.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import regex

log = logging.getLogger(__name__)

GENRES_BY_SOURCE = {
    "Spacetoon Songs": "Children", "Mama Makes Bread": "Children", "Green Library": "Children",
    "Majed": "Children", "chatGPT": "Children", "Kashkul": "Children",
    "ArabicMMLU": "Education", "ZAEBUC": "Education", "ALC": "Education", "BTEC": "Education",
    "Curriculum": "Education",
    "Sara": "Literature", "Arabian Nights": "Literature", "Hayy ibn Yaqdhan": "Literature",
    "Kalima": "Literature", "Hindawi": "Literature",
    "Subtitles": "Media", "QALB": "Media", "WikiNews": "Media",
    "Suleiman Al Issa": "Poetry", "My Language Sings": "Poetry", "Hanada Final Poems": "Poetry",
    "Hanging Odes": "Poetry",
    "UN DEC HR": "Politics", "Constitutions": "Politics",
    "Old Testament": "Religion", "Hadith": "Religion", "New Testament": "Religion", "Quran": "Religion",
    "Wikipedia": "Wiki",
}

SENT_MARKER_SOURCES = {"Quran", "New Testament", "Old Testament", "ArabicMMLU", "Constitutions"}

EOS_PNX_BUCKETS = {
    "comma": {",", "،"}, "period": {"."}, "colon": {":"}, "semicolon": {";", "؛"},
    "exclamation": {"!"}, "question": {"?", "؟"},
}
EOS_BUCKET_OF = {ch: name for name, chars in EOS_PNX_BUCKETS.items() for ch in chars}

_PNX_WORD = regex.compile(r"^[\p{P}\p{S}]+$")
_PNX_CHARS = regex.compile(r"[\p{P}\p{S}]+")
_MCQ_PREFIX = re.compile(r"^\s*\(\s*[ء-ي]+\s*\)\s*")
_SEG_TOKEN = re.compile(r"<\s*SEG\s*>")
_CONST_PREFIX = re.compile(
    r"^\s*(?:المادة\s*\d+\s*\.?|مادة\s*\(\s*\d+\s*\)|الفصل\s*\d+|\d+\s*\.)\s*"
)


@dataclass
class Sentence:
    id: int
    text: str


@dataclass
class Paragraph:
    id: int
    sentences: list[Sentence]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class Document:
    name: str
    source: str
    genre: str
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.paragraphs)


def is_pnx(word: str) -> bool:
    """True if every character of the word is punctuation or a symbol."""
    return bool(_PNX_WORD.match(word.strip()))


def remove_pnx(text: str) -> str:
    return _PNX_CHARS.sub("", text)


def remove_sent_markers(text: str, source: str) -> str:
    """Strip source-specific structural markers from one sentence."""
    if source == "Quran":
        parts = text.split(" ")
        return " ".join(parts[:-3]) if "(" in text and len(parts) > 3 else text
    if source in ("New Testament", "Old Testament"):
        parts = text.split(" ")
        return " ".join(parts[1:] if parts and parts[0].isdigit() else parts).strip()
    if source == "ArabicMMLU":
        return _MCQ_PREFIX.sub("", text).strip()
    if source == "Constitutions":
        return _CONST_PREFIX.sub("", text).strip()
    return text


def transform(text: str, source: str, setting: str, remove_markers: bool) -> str:
    text = _SEG_TOKEN.sub(" ", text)
    if remove_markers and source in SENT_MARKER_SOURCES:
        text = remove_sent_markers(text, source)
    if setting == "NoPnx":
        text = remove_pnx(text)
    return " ".join(text.split())


def load_documents(
    path: str | Path,
    split: str = "dev",
    setting: str = "Pnx",
    remove_sent_markers: bool | None = None,
    task: str = "NP",
) -> list[Document]:
    """Load documents for one split.

    ``path`` is the Hugging Face dataset id (``MBZUAI/AraSEG``), a local AraSeg TSV, or a ``.json``
    subset file from ``data/create_subsets.py`` (``split`` is ignored and the text is used as saved).

    From the Hub, the subset matching ``setting`` and ``task`` (``NP``, ``PA``, ``NoPnx-NP``,
    ``NoPnx-PA``) is loaded as published. From a TSV, ``setting`` "NoPnx" removes all
    punctuation/symbol characters. ``remove_sent_markers`` strips verse numbers / MCQ labels etc.;
    it defaults to True for NoPnx, which the NoPnx subsets already have applied.
    """
    if setting not in ("Pnx", "NoPnx"):
        raise ValueError(f"setting must be Pnx or NoPnx, got {setting!r}")
    if remove_sent_markers is None:
        remove_sent_markers = setting == "NoPnx"

    path = Path(path)
    is_subset = path.suffix == ".json"
    if is_subset:
        rows = load_subset_json(path)
    elif not path.exists():
        return load_hub_documents(str(path), split, setting, task, remove_sent_markers and setting == "Pnx")
    else:
        df = pd.read_csv(path, sep="\t")
        df = df[df["new_split"] == split]
        genre_col = next((c for c in ("genre", "Genre") if c in df.columns), None)
        rows = [
            (r.doc_name, r.source, getattr(r, genre_col) if genre_col else "",
             int(r.paragraph_id), int(r.sent_id_in_paragraph), str(r.text))
            for r in df.itertuples(index=False)
        ]

    grouped: dict[str, dict] = {}
    for doc_name, source, genre, para_id, sent_id, text in rows:
        text = " ".join(text.split()) if is_subset else transform(text, source, setting, remove_sent_markers)
        if not text:
            continue
        doc = grouped.setdefault(str(doc_name), {"source": source, "genre": genre, "sents": []})
        if doc["source"] != source:
            raise ValueError(f"Document {doc_name!r} has multiple sources")
        doc["sents"].append((para_id, sent_id, text))

    docs = []
    for name in sorted(grouped):
        g = grouped[name]
        paragraphs: dict[int, list[Sentence]] = {}
        for para_id, sent_id, text in sorted(g["sents"], key=lambda x: (x[0], x[1])):
            paragraphs.setdefault(para_id, []).append(Sentence(sent_id, text))
        tsv_genre = g["genre"] if isinstance(g["genre"], str) else ""
        genre = tsv_genre if is_subset else (GENRES_BY_SOURCE.get(g["source"]) or tsv_genre)
        docs.append(Document(name, g["source"], genre, [Paragraph(i, s) for i, s in paragraphs.items()]))

    if not docs:
        log.warning("No documents for split=%r in %s", split, path)
    log.info("Loaded %d docs from %s (split=%s, setting=%s, remove_markers=%s)",
             len(docs), path, split, setting, remove_sent_markers)
    return docs


def load_hub_documents(repo: str, split: str, setting: str, task: str, remove_markers: bool) -> list[Document]:
    """Documents of one split of the Hub subset for ``setting`` and ``task``, used as published."""
    from datasets import load_dataset

    if task not in ("NP", "PA"):
        raise ValueError(f"task must be NP or PA, got {task!r}")
    config = task if setting == "Pnx" else f"NoPnx-{task}"
    docs = []
    for r in load_dataset(repo, config, split=split):
        sents: dict[int, dict[int, list[str]]] = {}
        for tok, para_id, sent_id in zip(r["tokens"], r["paragraph_ids"], r["sent_ids"]):
            if sent_id >= 0:
                sents.setdefault(para_id, {}).setdefault(sent_id, []).append(tok)
        paragraphs = []
        for para_id, by_sent in sents.items():
            sentences = [Sentence(i, " ".join(t)) for i, t in by_sent.items()]
            if remove_markers:
                sentences = [Sentence(x.id, " ".join(remove_sent_markers(x.text, r["source"]).split()))
                             for x in sentences]
                sentences = [x for x in sentences if x.text]
            if sentences:
                paragraphs.append(Paragraph(para_id, sentences))
        docs.append(Document(r["doc_name"], r["source"], r["genre"], paragraphs))
    docs.sort(key=lambda d: d.name)
    log.info("Loaded %d docs from %s/%s (split=%s, remove_markers=%s)", len(docs), repo, config, split, remove_markers)
    return docs


def load_subset_json(path: Path) -> list[tuple]:
    """Read a subset file written by the subset sampler (list of serialized Documents)."""
    rows = []
    for d in json.loads(Path(path).read_text(encoding="utf-8")):
        for p in d["paragraphs"]:
            for s in p["sentences"]:
                rows.append((d["doc_name"], d["source"], d.get("genre", ""),
                             int(p["paragraph_id"]), int(s["sent_id_in_paragraph"]), s["text"]))
    return rows


def word_table(docs: list[Document]) -> pd.DataFrame:
    """One row per word, in document order, with gold sentence-boundary labels and slice metadata."""
    rows = []
    for doc in docs:
        for para in doc.paragraphs:
            for sent in para.sentences:
                words = sent.text.split()
                for wi, w in enumerate(words):
                    rows.append((
                        doc.name, doc.source, doc.genre, para.id, sent.id, w,
                        "Pnx" if is_pnx(w) else "Word",
                        EOS_BUCKET_OF.get(w.strip()),
                        int(wi == len(words) - 1),
                    ))
    return pd.DataFrame(rows, columns=[
        "doc_name", "source", "genre", "paragraph_id", "sent_id_in_paragraph", "word",
        "word_type", "eos_pnx_bucket", "gold",
    ])


def sentence_labels(sentences: list[str]) -> tuple[list[str], list[int]]:
    """Words of the sentences and binary labels (1 = last word of a sentence)."""
    words, labels = [], []
    for s in sentences:
        toks = s.split()
        words += toks
        labels += [0] * (len(toks) - 1) + [1] if toks else []
    return words, labels


def labels_to_segments(words: list[str], labels) -> list[str]:
    """Group words into sentences; a label of 1 closes the current sentence."""
    segments, cur = [], []
    for w, lab in zip(words, labels):
        cur.append(w)
        if lab == 1:
            segments.append(" ".join(cur))
            cur = []
    if cur:
        segments.append(" ".join(cur))
    return segments
