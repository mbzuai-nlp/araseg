"""Sample training subsets for the data-size and genre experiments (paper §6.2, §6.3).

    python -m data.create_subsets configs/subsets.yaml experiment=held_out_genre

Selection is document-level and the same documents are chosen for NP and PA; only the
structure differs (NP merges each document into a single paragraph). Each subset is written to

    {output_root}/{experiment}/{NP|PA}/seed_{seed}/{tag}/
        train_documents.json  dev_documents.json  test_documents.json  manifest.json

Experiments (tag):
    overall_curve   x% of the training words, sampled within each genre in train-split proportions;
                    full dev/test                                                    (pct_10%)
    held_out_genre  all training documents except one genre; dev of the other genres,
                    test of the held-out genre (genre transferability)               (heldout_Media)
    single_genre    only the training documents of one genre; dev/test of that genre
                    (cross-genre benefit)                                            (genre_Media)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from utils.config import cli
from utils.data import Document, Paragraph, Sentence, load_documents

log = logging.getLogger(__name__)


def n_words(doc: Document) -> int:
    return sum(len(s.text.split()) for p in doc.paragraphs for s in p.sentences)


def genre_seed(seed: int, genre: str) -> int:
    return seed + sum((i + 1) * ord(ch) for i, ch in enumerate(genre))


def shuffled(docs: list[Document], seed: int) -> list[Document]:
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    return docs


def by_genre(docs: list[Document]) -> dict[str, list[Document]]:
    out: dict[str, list[Document]] = {}
    for d in docs:
        out.setdefault(d.genre, []).append(d)
    return out


def prefix_until(docs: list[Document], target: int) -> list[Document]:
    """Shortest prefix of whole documents reaching ``target`` words."""
    out, total = [], 0
    for d in docs:
        out.append(d)
        total += n_words(d)
        if total >= target:
            break
    return out


def as_single_paragraph(doc: Document) -> Document:
    sents = [Sentence(i, s.text) for i, s in enumerate(s for p in doc.paragraphs for s in p.sentences)]
    return Document(doc.name, doc.source, doc.genre, [Paragraph(0, sents)])


def to_json(docs: list[Document]) -> list[dict]:
    return [{
        "doc_name": d.name, "source": d.source, "genre": d.genre, "document_word_count": n_words(d),
        "paragraphs": [{"paragraph_id": p.id, "paragraph_word_count": sum(len(s.text.split()) for s in p.sentences),
                        "sentences": [{"sent_id_in_paragraph": s.id, "text": s.text,
                                       "sentence_word_count": len(s.text.split())} for s in p.sentences]}
                       for p in d.paragraphs],
    } for d in docs]


def summary(docs: list[Document]) -> dict:
    words, counts = {}, {}
    for d in docs:
        words[d.genre] = words.get(d.genre, 0) + n_words(d)
        counts[d.genre] = counts.get(d.genre, 0) + 1
    return {"num_docs": len(docs), "num_words": sum(words.values()), "doc_names": [d.name for d in docs],
            "genre_word_counts": dict(sorted(words.items())), "genre_doc_counts": dict(sorted(counts.items()))}


def build(experiment: str, seed: int, train: list[Document], dev: list[Document], test: list[Document],
          cfg: dict) -> list[tuple[str, list, list, list, dict]]:
    """All subsets of one experiment for one seed: (tag, train, dev, test, metadata)."""
    train_g, dev_g, test_g = by_genre(train), by_genre(dev), by_genre(test)
    genres = sorted(train_g)
    shuf_g = {g: shuffled(train_g[g], genre_seed(seed, g)) for g in genres}
    out = []

    if experiment == "overall_curve":
        words_g = {g: sum(map(n_words, docs)) for g, docs in train_g.items()}
        total = sum(words_g.values())
        for pct in cfg["percentages"]:
            budget = max(1, int(round(total * pct)))
            sel = [d for g in genres for d in prefix_until(shuf_g[g], max(1, int(round(budget * words_g[g] / total))))]
            meta = {"nominal_percentage_of_train": pct, "realized_total_words": sum(map(n_words, sel))}
            out.append((f"pct_{str(int(round(pct * 100))).zfill(2)}%", sel, dev, test, meta))

    elif experiment in ("held_out_genre", "single_genre"):
        for g in cfg.get("genres") or genres:
            if g not in train_g:
                raise ValueError(f"unknown genre {g!r}; have {genres}")
            if experiment == "held_out_genre":
                seen = [d for h in genres if h != g for d in shuf_g[h]]
                seen_dev = [d for h in genres if h != g for d in dev_g.get(h, [])]
                out.append((f"heldout_{g}", seen, seen_dev, test_g.get(g, []), {"held_out_genre": g}))
            else:
                out.append((f"genre_{g}", shuf_g[g], dev_g.get(g, []), test_g.get(g, []), {"target_genre": g}))
    else:
        raise ValueError(f"unknown experiment {experiment!r}; use overall_curve, held_out_genre or single_genre")
    return out


def main() -> None:
    cfg = cli(__doc__.splitlines()[1])
    data = cfg["data"]
    root, experiment = Path(cfg["output_root"]), cfg["experiment"]
    for task in cfg.get("tasks", ["NP", "PA"]):
        splits = {s: load_documents(data["path"], s, data.get("setting", "Pnx"), data.get("remove_sent_markers"), task)
                  for s in ("train", "dev", "test")}
        for seed in cfg["seeds"]:
            for tag, train, dev, test, meta in build(experiment, seed, splits["train"], splits["dev"], splits["test"], cfg):
                out = root / experiment / task / f"seed_{seed}" / tag
                if out.exists() and not cfg.get("overwrite"):
                    continue
                out.mkdir(parents=True, exist_ok=True)
                parts = {"train": train, "dev": dev, "test": test}
                if task == "NP":
                    parts = {k: [as_single_paragraph(d) for d in v] for k, v in parts.items()}
                manifest = {"metadata": {"experiment": experiment, "setting": task, "seed": seed,
                                         "data": data, **meta},
                            "setting": task, **{f"{k}_summary": summary(v) for k, v in parts.items()}}
                (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
                for k, v in parts.items():
                    (out / f"{k}_documents.json").write_text(json.dumps(to_json(v), ensure_ascii=False, indent=2))
                log.info("%s: %d train docs, %d words", out, len(train), sum(map(n_words, train)))


if __name__ == "__main__":
    main()
