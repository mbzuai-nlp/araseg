"""CAMeL-BERT token classification: subtoken alignment, windowing, training pieces and the
``CamelBERTSegmenter`` used for evaluation and the driver.

Each word's label sits on its last subtoken.
Long sequences are cut into windows of ``max_length`` tokens ([CLS] + span + [SEP]).
At inference, windows overlap by ``stride`` tokens and overlapping logits are averaged.
A word is a sentence end when argmax == 1 (binary models: "boundary"; pnx-insertion models: "\\n").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

from utils.segmenter import Segmenter

log = logging.getLogger(__name__)

IGNORE = -100


def tokenize_words(tokenizer, words: list[str]) -> tuple[list[int], list[int | None]]:
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    return enc["input_ids"], enc.word_ids()


def last_subtokens(word_ids: list[int | None]) -> dict[int, int]:
    """Word index -> index of its last subtoken."""
    return {w: t for t, w in enumerate(word_ids) if w is not None}


def word_to_token_labels(word_labels: list[int], word_ids: list[int | None]) -> list[int]:
    out = [IGNORE] * len(word_ids)
    for w, t in last_subtokens(word_ids).items():
        out[t] = word_labels[w]
    return out


def token_to_word_logits(logits: np.ndarray, word_ids: list[int | None], n_words: int) -> np.ndarray:
    out = np.zeros((n_words, logits.shape[-1]), dtype=np.float32)
    for w, t in last_subtokens(word_ids).items():
        out[w] = logits[t]
    return out


def make_windows(record: dict, tokenizer, max_length: int, step: int) -> list[dict]:
    """Training windows for one word record {"words", "labels"}.

    Consecutive windows start ``step`` tokens apart (so they overlap by max_length - 2 - step).
    """
    ids, word_ids = tokenize_words(tokenizer, record["words"])
    labels = word_to_token_labels(record["labels"], word_ids)
    cls, sep, inner = tokenizer.cls_token_id, tokenizer.sep_token_id, max_length - 2
    windows, start = [], 0
    while start < len(ids):
        end = min(start + inner, len(ids))
        windows.append({"input_ids": [cls] + ids[start:end] + [sep], "attention_mask": [1] * (end - start + 2),
                        "labels": [IGNORE] + labels[start:end] + [IGNORE]})
        if end == len(ids):
            break
        start += step
    return windows


@dataclass
class PadCollator:
    max_length: int
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        def pad(key, value):
            return torch.tensor([f[key] + [value] * (self.max_length - len(f[key])) for f in features])

        return {"input_ids": pad("input_ids", self.pad_token_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", IGNORE)}


class WeightedTrainer(Trainer):
    """Cross-entropy with optional class weights."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                           weight=weight, ignore_index=IGNORE)
        return (loss, outputs) if return_outputs else loss


def _prf(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = int((gold & pred).sum())
    fp, fn = int((~gold & pred).sum()), int((gold & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def compute_dev_metrics(eval_pred) -> dict:
    """Window-level dev metrics during training. ``f1`` = sentence-boundary F1 (argmax == 1)."""
    logits, labels = eval_pred
    pred, labels = logits.argmax(-1).reshape(-1), labels.reshape(-1)
    mask = labels != IGNORE
    pred, labels = pred[mask], labels[mask]
    p, r, f = _prf(labels == 1, pred == 1)
    out = {"precision": p, "recall": r, "f1": f}
    if logits.shape[-1] > 2:
        from sklearn.metrics import f1_score

        pnx = labels >= 2
        classes = list(range(2, logits.shape[-1]))
        out["pnx_macro_f1"] = float(f1_score(labels[pnx], pred[pnx], labels=classes, average="macro",
                                             zero_division=0)) if pnx.any() else 0.0
    return out


def strided_logits(model, tokenizer, docs_ids: list[list[int]], max_length: int, stride: int,
                   batch_size: int, device) -> list[np.ndarray]:
    """Token logits for each sequence; windows overlap by ``stride`` tokens and are averaged."""
    inner = max_length - 2
    if not 0 <= stride < inner:
        raise ValueError(f"stride must be in [0, {inner})")
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    n_labels = model.config.num_labels
    sums = [np.zeros((len(ids), n_labels), np.float32) for ids in docs_ids]
    counts = [np.zeros(len(ids), np.float32) for ids in docs_ids]
    windows = []
    for d, ids in enumerate(docs_ids):
        start = 0
        while start < len(ids):
            end = min(start + inner, len(ids))
            windows.append((d, start, end))
            if end == len(ids):
                break
            start += inner - stride
    model.eval()
    for b in range(0, len(windows), batch_size):
        batch = windows[b: b + batch_size]
        ids = [[cls] + docs_ids[d][s:e] + [sep] for d, s, e in batch]
        mask = [[1] * len(x) + [0] * (max_length - len(x)) for x in ids]
        ids = [x + [pad] * (max_length - len(x)) for x in ids]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(ids, device=device),
                           attention_mask=torch.tensor(mask, device=device)).logits.float().cpu().numpy()
        for (d, s, e), lg in zip(batch, logits):
            sums[d][s:e] += lg[1: 1 + e - s]
            counts[d][s:e] += 1
    return [s / np.maximum(c, 1)[:, None] for s, c in zip(sums, counts)]


class CamelBERTSegmenter(Segmenter):
    def __init__(self, model_path: str, max_length: int = 512, stride: int = 64, batch_size: int = 16,
                 device: str | None = None):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device).eval()
        self.max_length, self.stride, self.batch_size = max_length, stride, batch_size

    def predict(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        words = [t.split() for t in texts]
        encoded = [tokenize_words(self.tokenizer, w) if w else ([], []) for w in words]
        logits = strided_logits(self.model, self.tokenizer, [ids for ids, _ in encoded], self.max_length,
                                self.stride, self.batch_size, self.device)
        out = []
        for w, (_, word_ids), lg in zip(words, encoded, logits):
            wl = token_to_word_logits(lg, word_ids, len(w))
            probs = np.exp(wl - wl.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            out.append(((wl.argmax(-1) == 1).astype(int).tolist(), probs[:, 1].tolist()))
        return out
