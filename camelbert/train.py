"""Fine-tune CAMeL-BERT for sentence segmentation, then evaluate the result.

    python -m camelbert.train configs/train/camelbert.yaml [key=value ...]

Outputs in {output_dir}/:
    final/             best checkpoint (plain HF model + tokenizer), loadable by the segmenter
    eval/{split}/      evaluation outputs (see evaluation/evaluate.py)
    checkpoints/       HF Trainer checkpoints

With data.subset_dir set (a directory written by data.create_subsets), train/dev/test documents come
from its {split}_documents.json files instead of data.path.

Label schemes (``labels``):
    boundary       binary: 1 on the last word of each sentence
    pnx_insertion  WtP-style, 11 classes. Punctuation-only tokens are removed from the input and
                   each word is labelled with what follows it: 1 = sentence end ("\\n"),
                   2+ = a punctuation mark, 0 = nothing. Evaluate it on NoPnx text (CAMeLBERT-PnxInsert).

Variants from the paper:
    boundary_mode: pnx / word   train only on punctuation-based / word-based boundaries
                                (CAMeLBERT-Pnx / CAMeLBERT-Word)
    pnx_dropout: p              drop each punctuation token with probability p, resampled every epoch
                                (CAMeLBERT-PnxDrop)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, set_seed

from camelbert.model import CamelBERTSegmenter, PadCollator, WeightedTrainer, compute_dev_metrics, make_windows
from utils.config import cli, save_config
from utils.data import Document, is_pnx, load_documents, sentence_labels
from evaluation.evaluate import run_eval

log = logging.getLogger(__name__)

PNX_LABELS = {"\n": 1, ".": 2, ",": 3, "،": 3, ":": 4, ";": 5, "؛": 5, "?": 6, "؟": 6, "!": 7, '"': 8, "(": 9, ")": 10}
PNX_NAMES = {0: "Word", 1: "\\n", 2: ".", 3: ",", 4: ":", 5: ";", 6: "?", 7: "!", 8: '"', 9: "(", 10: ")"}
BOUNDARY_PNX_CHARS = set(".!?:;,،؟؛")


def pnx_insertion_labels(tokens: list[str]) -> tuple[list[str], list[int]]:
    """Drop punctuation tokens; label each kept word with the class of the token after it."""
    words, labels = [], []
    for i, tok in enumerate(tokens):
        if is_pnx(tok):
            continue
        words.append(tok)
        labels.append(PNX_LABELS.get(tokens[i + 1], 0) if i + 1 < len(tokens) else 1)
    if words:
        labels[-1] = 1
    return words, labels


def make_records(docs: list[Document], task: str, scheme: str) -> list[dict]:
    """One record per document (NP) or paragraph (PA): words and labels."""
    def encode(sentences):
        if scheme == "boundary":
            return sentence_labels(sentences)
        words, labels = [], []
        for s in sentences:
            w, lab = pnx_insertion_labels(s.split())
            words += w
            labels += lab
        return words, labels

    records = []
    for doc in docs:
        if task == "PA":
            for p in doc.paragraphs:
                words, labels = encode([s.text for s in p.sentences])
                if words:
                    records.append({"words": words, "labels": labels})
        else:
            words, labels = encode([s.text for p in doc.paragraphs for s in p.sentences])
            if words:
                records.append({"words": words, "labels": labels})
    return records


def filter_boundaries(records: list[dict], mode: str) -> list[dict]:
    """boundary_mode: keep all positives, only those on punctuation-final words (pnx), or the rest (word)."""
    if mode not in ("all", "pnx", "word"):
        raise ValueError(f"boundary_mode must be all, pnx or word, got {mode!r}")
    if mode == "all":
        return records
    out = []
    for r in records:
        labels = [0 if lab == 1 and ((w[-1] in BOUNDARY_PNX_CHARS) != (mode == "pnx")) else lab
                  for w, lab in zip(r["words"], r["labels"])]
        out.append({**r, "labels": labels})
    return out


def pnx_dropout(records: list[dict], prob: float, rng: np.random.Generator) -> list[dict]:
    """Drop each punctuation token with probability ``prob``; its positive label moves to the previous kept word."""
    out, n_pnx, n_dropped = [], 0, 0
    for r in records:
        words, labels = [], []
        for w, lab in zip(r["words"], r["labels"]):
            if is_pnx(w):
                n_pnx += 1
                if rng.random() < prob:
                    n_dropped += 1
                    if words:
                        labels[-1] = max(labels[-1], lab)
                    continue
            words.append(w)
            labels.append(lab)
        if words:
            out.append({"words": words, "labels": labels})
    log.info("pnx dropout: dropped %d / %d punctuation tokens", n_dropped, n_pnx)
    return out


def class_weights(records: list[dict], spec, num_labels: int) -> torch.Tensor | None:
    """``balanced`` (inverse frequency, normalized to sum 1), ``none``, or an explicit list."""
    if spec in (None, "none"):
        return None
    if isinstance(spec, (list, tuple)):
        return torch.tensor(spec, dtype=torch.float32)
    if spec not in ("balanced", "auto"):
        raise ValueError(f"class_weights must be balanced, none or a list, got {spec!r}")
    counts = np.bincount([lab for r in records for lab in r["labels"]], minlength=num_labels).astype(float)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    return torch.tensor(inv / inv.sum(), dtype=torch.float32)


def main() -> None:
    cfg = cli(__doc__.splitlines()[1])
    out_dir = Path(cfg["output_dir"])
    save_config(cfg, out_dir)
    set_seed(cfg["training"]["seed"])
    data, m, t = cfg["data"], cfg["model"], cfg["training"]
    task, scheme = cfg["task"], cfg["labels"]
    num_labels = 2 if scheme == "boundary" else len(PNX_NAMES)

    subset = Path(data["subset_dir"]) if data.get("subset_dir") else None

    def load(split, setting=data["setting"], markers=data.get("remove_sent_markers")):
        if subset:
            return load_documents(subset / f"{split}_documents.json")
        return load_documents(data["path"], split, setting, markers, task)

    train_docs = load(data["train_split"])
    dev_docs = load(data["dev_split"])
    if cfg.get("limit_docs"):
        train_docs, dev_docs = train_docs[: cfg["limit_docs"]], dev_docs[: cfg["limit_docs"]]

    train_records = filter_boundaries(make_records(train_docs, task, scheme), cfg.get("boundary_mode", "all"))
    dev_records = make_records(dev_docs, task, scheme)

    tokenizer = AutoTokenizer.from_pretrained(m["name"])

    def windows(records):
        return [w for r in records for w in make_windows(r, tokenizer, m["max_length"], m["window_step"])]

    epochs = t["epochs"]
    if cfg.get("pnx_dropout"):
        train_windows = [w for e in range(int(epochs)) for w in windows(
            pnx_dropout(train_records, cfg["pnx_dropout"], np.random.default_rng(t["seed"] + e)))]
        epochs = 1
    else:
        train_windows = windows(train_records)
    weights = class_weights(train_records, t.get("class_weights"), num_labels)
    log.info("train windows=%d dev windows from %d records; class weights=%s", len(train_windows),
             len(dev_records), weights)

    model = AutoModelForTokenClassification.from_pretrained(
        m["name"], num_labels=num_labels,
        id2label={i: (PNX_NAMES[i] if scheme == "pnx_insertion" else ["O", "EOS"][i]) for i in range(num_labels)})
    if cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            learning_rate=t["learning_rate"],
            warmup_steps=t["warmup"],
            weight_decay=t["weight_decay"],
            gradient_accumulation_steps=t["grad_accum"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_steps=t["eval_steps"],
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            label_names=["labels"],
            remove_unused_columns=False,
            report_to=["wandb"] if cfg.get("wandb_project") else ["none"],
            seed=t["seed"],
        ),
        train_dataset=Dataset.from_list(train_windows),
        eval_dataset=Dataset.from_list(windows(dev_records)),
        data_collator=PadCollator(m["max_length"], tokenizer.pad_token_id),
        compute_metrics=compute_dev_metrics,
        class_weights=weights,
    )
    trainer.train()

    final = out_dir / "final"
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    log.info("saved %s", final)

    ev = cfg.get("final_eval") or {}
    setting = ev.get("setting") or ("NoPnx" if scheme == "pnx_insertion" else data["setting"])
    markers = ev.get("remove_sent_markers", data.get("remove_sent_markers") if setting == data["setting"] else None)
    segmenter = CamelBERTSegmenter(str(final), max_length=m["max_length"], stride=m["stride"],
                                   batch_size=t["batch_size"])
    for split in ev.get("splits", ["dev"]):
        docs = load(split, setting, markers)
        if cfg.get("limit_docs"):
            docs = docs[: cfg["limit_docs"]]
        run_eval(segmenter, docs, task, out_dir / "eval" / split)


if __name__ == "__main__":
    main()
