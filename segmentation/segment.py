"""Segmentation driver: split documents from a JSON file into sentences with any segmenter.

    python -m segmentation.segment configs/segment.yaml input_json=docs.json experiment_name=demo
    python -m segmentation.segment configs/segment.yaml input_json=docs.json experiment_name=demo \\
        segmenter.name=camelbert segmenter.model_path=results/camelbert/Pnx_NP/final

Input JSON, top-level keys are split names (a flat {doc_id: {...}} map is also accepted):

    {"dev": {"doc_id": {"doc_text": "..."}, ...}, "test": {...}}

Output per split at {output_dir}/{experiment_name}/{split}/{output_file_name}:

    {"doc_id": {"sentences": ["...", ...]}, ...}

With return_word_details=true each entry also has per-word "predictions" (1 = sentence end) and,
for CAMeL-BERT, "probabilities". With task=PA each line of doc_text is segmented on its own.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from utils.config import cli, save_config
from utils.data import labels_to_segments
from utils.segmenter import build_segmenter

log = logging.getLogger(__name__)


def main() -> None:
    cfg = cli(__doc__.splitlines()[1])
    if not cfg.get("input_json") or not cfg.get("experiment_name"):
        raise SystemExit("set input_json=... and experiment_name=...")
    data = json.loads(Path(cfg["input_json"]).read_text(encoding="utf-8"))
    if all(isinstance(v, dict) and "doc_text" in v for v in data.values()):
        data = {"docs": data}
    split = cfg.get("split", "dev")
    splits = list(data) if split == "all" else [split]
    missing = [s for s in splits if s not in data]
    if missing:
        raise SystemExit(f"split(s) {missing} not in input; available: {list(data)}")

    segmenter = build_segmenter(cfg["segmenter"])
    out_root = Path(cfg.get("output_dir", "outputs")) / cfg["experiment_name"]
    save_config(cfg, out_root)
    for split in splits:
        ids = list(data[split])
        texts = [" ".join(data[split][i]["doc_text"].split()) for i in ids]
        if cfg.get("task", "NP") == "PA":
            paragraphs = [[" ".join(line.split()) for line in data[split][i]["doc_text"].split("\n") if line.strip()]
                          for i in ids]
            preds = [(sum((lab for lab, _ in doc), []),
                      None if any(p is None for _, p in doc) else sum((p for _, p in doc), []))
                     for doc in segmenter.predict_paragraphs(paragraphs)]
        else:
            preds = segmenter.predict(texts)

        out = {}
        for doc_id, text, (labels, probs) in zip(ids, texts, preds):
            entry = {"sentences": labels_to_segments(text.split(), labels)}
            if cfg.get("return_word_details"):
                entry["predictions"] = [int(x) for x in labels]
                if probs is not None:
                    entry["probabilities"] = [float(x) for x in probs]
            out[doc_id] = entry
        path = out_root / split / cfg.get("output_file_name", "segments.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8")
        log.info("saved %d documents to %s", len(out), path)


if __name__ == "__main__":
    main()
