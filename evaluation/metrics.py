"""Precision / recall / F1 for sentence boundaries: overall, by genre, by word type (boundaries after
punctuation vs after words) and by end-of-sentence punctuation mark.

Every slice is reported as ``doc_macro`` (mean of per-document scores, the paper's main metric)
and ``micro`` (pooled over all words). Input is the word table from ``data.word_table`` with a
``pred`` column.
"""
from __future__ import annotations

import pandas as pd

from utils.data import EOS_PNX_BUCKETS


def _scores(c: pd.DataFrame) -> pd.DataFrame:
    """Per-row precision/recall/F1 from tp/fp/fn count columns (0 when undefined)."""
    return pd.DataFrame({
        "precision": (c.tp / (c.tp + c.fp)).fillna(0.0),
        "recall": (c.tp / (c.tp + c.fn)).fillna(0.0),
        "f1": (2 * c.tp / (2 * c.tp + c.fp + c.fn)).fillna(0.0),
    })


def _micro_macro(df: pd.DataFrame) -> dict:
    per_doc = df.groupby("doc_name")[["tp", "fp", "fn"]].sum()
    total = per_doc.sum().to_frame().T
    micro = _scores(total).iloc[0].to_dict()
    micro["support"] = int(total.tp.iloc[0] + total.fn.iloc[0])
    macro = _scores(per_doc).mean().to_dict() if len(per_doc) else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"micro": micro, "doc_macro": macro}


def _by(df: pd.DataFrame, col: str) -> dict:
    return {str(k): _micro_macro(g) for k, g in df.groupby(col, sort=True)}


def compute_metrics(table: pd.DataFrame) -> dict:
    gold, pred = table.gold.astype(bool), table.pred.astype(bool)
    df = table.assign(tp=(gold & pred).astype(int), fp=(~gold & pred).astype(int), fn=(gold & ~pred).astype(int))
    m = {
        "overall": _micro_macro(df),
        "by_genre": _by(df, "genre"),
        "by_word_type": _by(df, "word_type"),
        "eos_pnx": {},
    }
    for bucket in EOS_PNX_BUCKETS:
        b = df[df.eos_pnx_bucket == bucket]
        m["eos_pnx"][bucket] = {"overall": _micro_macro(b), "by_genre": _by(b, "genre")}
    return m


def flatten(metrics: dict) -> pd.DataFrame:
    """Long table: slice, group, aggregation, precision, recall, f1, support."""
    rows = []

    def walk(node, path):
        if "micro" in node:
            for agg in ("micro", "doc_macro"):
                rows.append({"slice": "/".join(path[:-1]) or path[0], "group": path[-1] if len(path) > 1 else "",
                             "aggregation": agg, **node[agg]})
            return
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, path + [k])

    for key in ("overall", "by_genre", "by_word_type", "eos_pnx"):
        walk(metrics[key], [key])
    return pd.DataFrame(rows)
