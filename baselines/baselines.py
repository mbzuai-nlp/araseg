"""Baselines from the paper: Paragraph Breaks, NLTK Punkt, SpaCySENT, Ersatz, PySBD and SaT."""
from __future__ import annotations

import logging
import pickle
import subprocess
from pathlib import Path

from utils.data import load_documents
from utils.segmenter import FunctionSegmenter, Segmenter

log = logging.getLogger(__name__)


def paragraph_breaks() -> Segmenter:
    """Paragraph Breaks: every paragraph is one sentence (PA only)."""
    return FunctionSegmenter(lambda t: [t], tasks=("PA",))


def nltk_punkt(train_data: str, train_split: str = "train", setting: str = "Pnx",
               cache_path: str | None = None) -> Segmenter:
    """NLTK Punkt trained on AraSeg gold sentences (one per line, blank line between paragraphs)."""
    from nltk.tokenize.punkt import PunktSentenceTokenizer

    if cache_path and Path(cache_path).exists():
        tok = pickle.loads(Path(cache_path).read_bytes())
    else:
        docs = load_documents(train_data, split=train_split, setting=setting, task="PA")
        corpus = "\n\n".join("\n\n".join("\n".join(s.text for s in p.sentences) for p in d.paragraphs) for d in docs)
        tok = PunktSentenceTokenizer(train_text=corpus)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_bytes(pickle.dumps(tok))
    return FunctionSegmenter(tok.tokenize)


def spacy_sent() -> Segmenter:
    """SpaCySENT: the rule-based spaCy sentencizer on a blank Arabic pipeline."""
    import spacy

    nlp = spacy.blank("ar")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 10_000_000
    return FunctionSegmenter(lambda t: [str(s) for s in nlp(t).sents])


def ersatz(binary: str = "ersatz") -> Segmenter:
    def run(text: str) -> list[str]:
        res = subprocess.run([binary], input=text, capture_output=True, text=True, check=True)
        return res.stdout.split("\n")

    return FunctionSegmenter(run)


def pysbd() -> Segmenter:
    import pysbd as _pysbd

    seg = _pysbd.Segmenter(language="ar", clean=False)
    return FunctionSegmenter(seg.segment)


def sat(model: str = "sat-12l-sm", device: str = "auto") -> Segmenter:
    import torch
    from wtpsplit import SaT

    model_ = SaT(model)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_.to(device)
    if device == "cuda":
        model_.half()
    log.info("SaT %s on %s", model, device)
    return FunctionSegmenter(model_.split)


BASELINES = {"paragraph_breaks": paragraph_breaks, "nltk_punkt": nltk_punkt, "spacy_sent": spacy_sent,
             "ersatz": ersatz, "pysbd": pysbd, "sat": sat}
