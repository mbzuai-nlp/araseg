"""Evaluate segmenters on the AraSeg dataset.

    python -m evaluation.evaluate configs/eval/baselines.yaml [key=value ...]

Writes to {output_dir}/{setting}/{task}/{segmenter id}/:
    metrics.json     all slices (overall, by_genre, by_word_type, eos_pnx)
    metrics.tsv      the same, one row per slice/group/aggregation
    predictions.tsv  one row per word: metadata, gold, pred, prob
    llm_calls.jsonl  prompts and responses (LLM segmenters only)

Tasks: NP = the model sees each whole document as one stream; PA = each paragraph separately.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from utils.config import cli, save_config
from utils.data import Document, load_documents, word_table
from evaluation.metrics import compute_metrics, flatten
from utils.segmenter import Segmenter, build_segmenter

log = logging.getLogger(__name__)


def run_eval(segmenter: Segmenter, docs: list[Document], task: str, out_dir: str | Path | None = None) -> dict:
    """Segment ``docs`` under ``task`` (NP / PA), score against gold, optionally write outputs."""
    table = word_table(docs)
    if task == "NP":
        preds = segmenter.predict([d.text for d in docs])
    elif task == "PA":
        preds = [p for doc in segmenter.predict_paragraphs([[p.text for p in d.paragraphs] for d in docs]) for p in doc]
    else:
        raise ValueError(f"task must be NP or PA, got {task!r}")
    table["pred"] = np.concatenate([np.asarray(labels, dtype=int) for labels, _ in preds]) if preds else []
    if preds and all(p is not None for _, p in preds):
        table["prob"] = np.concatenate([np.asarray(p, dtype=float) for _, p in preds])
    metrics = compute_metrics(table)
    o = metrics["overall"]
    log.info("%s F1 micro=%.4f doc_macro=%.4f", task, o["micro"]["f1"], o["doc_macro"]["f1"])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        flatten(metrics).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)
        table.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
        if getattr(segmenter, "calls", None):
            with open(out_dir / "llm_calls.jsonl", "w", encoding="utf-8") as f:
                for call in segmenter.calls:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")
            segmenter.calls.clear()
        log.info("wrote %s", out_dir)
    return metrics


def main() -> None:
    cfg = cli(__doc__.splitlines()[1])
    data = cfg["data"]
    docs_by_task = {}

    def docs_for(task):
        if task not in docs_by_task:
            docs = load_documents(data["path"], data.get("split", "dev"), data.get("setting", "Pnx"),
                                  data.get("remove_sent_markers"), task)
            docs_by_task[task] = docs[: cfg["limit_docs"]] if cfg.get("limit_docs") else docs
        return docs_by_task[task]

    root = Path(cfg.get("output_dir", "results")) / data.get("setting", "Pnx")
    for seg_cfg in cfg["segmenters"]:
        seg_id = seg_cfg.get("id", seg_cfg["name"])
        todo = [t for t in cfg.get("tasks", ["NP", "PA"])
                if cfg.get("overwrite") or not (root / t / seg_id / "metrics.json").exists()]
        if not todo:
            log.info("skipping %s: results exist (set overwrite=true to redo)", seg_id)
            continue
        segmenter = build_segmenter(seg_cfg)
        for task in todo:
            if task not in segmenter.tasks:
                log.info("skipping %s / %s: not defined for this task", seg_id, task)
                continue
            log.info("=== %s / %s", seg_id, task)
            out = root / task / seg_id
            run_eval(segmenter, docs_for(task), task, out)
            save_config({**cfg, "segmenters": [seg_cfg], "tasks": [task]}, out)


if __name__ == "__main__":
    main()
