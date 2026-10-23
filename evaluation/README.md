# Evaluation

We evaluate with boundary-level precision (P), recall (R), and F<sub>1</sub>, where a prediction is correct only if it exactly matches a gold boundary. Scores are computed per document and macro-averaged over documents, to avoid bias toward larger documents.

`evaluate.py` runs every segmenter in a config ([configs/eval](../configs/eval)) on each task and writes to `{output_dir}/{setting}/{task}/{id}/`:

* `metrics.json`: P, R, and F<sub>1</sub> overall, by genre, for punctuation- and word-based boundaries (`by_word_type`), and per punctuation mark (`eos_pnx`). Every score is given as `doc_macro` (our main metric) and `micro` (pooled over all tokens).
* `metrics.tsv`: the same scores as one table.
* `predictions.tsv`: one row per token with its document, genre, gold label, and prediction.

```bash
python -m evaluation.evaluate configs/eval/baselines.yaml data.setting=Pnx data.split=test
```

Any value in a config can be overridden with `key=value`, e.g. `tasks=[PA]`, `limit_docs=5`, or `overwrite=true`.
