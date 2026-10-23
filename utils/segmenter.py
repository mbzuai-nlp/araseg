"""The segmenter interface shared by baselines, LLMs and CAMeLBERT, and ``build_segmenter`` to create one
from a config block (``name`` plus that segmenter's keyword arguments).

Every segmenter takes whitespace-tokenized text and returns sentences made of exactly the
input words (outputs that drift from the input are projected back with ``alignment.aligner.project``).

    seg = build_segmenter({"name": "sat"})
    seg.segment_many(["first doc text", "second doc text"])

Subclasses implement ``split(text) -> list[str]`` or override ``predict`` (labels + probs).
"""
from __future__ import annotations

from tqdm import tqdm

from alignment.aligner import project
from utils.data import labels_to_segments


class Segmenter:
    tasks = ("NP", "PA")

    def split(self, text: str) -> list[str]:
        """Raw backend output: a list of sentence strings for ``text``."""
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[tuple[list[int], list[float] | None]]:
        """Per text: one boundary label per word (1 = sentence ends here) and optional probabilities."""
        out = []
        for text in tqdm(texts, desc=type(self).__name__, disable=len(texts) < 2):
            words = text.split()
            out.append((project(self.split(" ".join(words)), words) if words else [], None))
        return out

    def predict_paragraphs(self, docs: list[list[str]]) -> list[list[tuple]]:
        """Like ``predict`` for documents given as paragraph lists; sentences never cross paragraphs."""
        flat = self.predict([p for doc in docs for p in doc])
        out, i = [], 0
        for doc in docs:
            out.append(flat[i: i + len(doc)])
            i += len(doc)
        return out

    def segment_many(self, texts: list[str]) -> list[list[str]]:
        return [labels_to_segments(t.split(), labels) for t, (labels, _) in zip(texts, self.predict(texts))]

    def segment(self, text: str) -> list[str]:
        return self.segment_many([text])[0]


class FunctionSegmenter(Segmenter):
    def __init__(self, fn, tasks=("NP", "PA")):
        self.fn = fn
        self.tasks = tasks

    def split(self, text: str) -> list[str]:
        return self.fn(text)


def build_segmenter(cfg: dict) -> Segmenter:
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "id")}
    name = cfg["name"]
    if name == "camelbert":
        from camelbert.model import CamelBERTSegmenter
        return CamelBERTSegmenter(**kwargs)
    if name in ("openai", "gemini"):
        from llms.llms import APISegmenter
        return APISegmenter(provider=name, **kwargs)
    if name == "hf_llm":
        from llms.llms import HFSegmenter
        return HFSegmenter(**kwargs)
    from baselines.baselines import BASELINES
    if name not in BASELINES:
        raise ValueError(f"unknown segmenter {name!r}; choose from "
                         f"{sorted(BASELINES) + ['camelbert', 'openai', 'gemini', 'hf_llm']}")
    return BASELINES[name](**kwargs)
