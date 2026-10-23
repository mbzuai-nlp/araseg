# Segmentation

`segment.py` segments your own documents with any of our models or baselines. The input is a JSON file whose top-level keys are split names:

```json
{"dev": {"doc_id": {"doc_text": "..."}}, "test": {"doc_id": {"doc_text": "..."}}}
```

```bash
python -m segmentation.segment configs/segment.yaml input_json=docs.json experiment_name=demo \
    segmenter.name=camelbert segmenter.model_path=MBZUAI/sentseg-camelbert-NP split=all
```

The sentences are written to `outputs/{experiment_name}/{split}/segments.json`:

```json
{"doc_id": {"sentences": ["...", "..."]}}
```

* `segmenter` takes any segmenter block from [configs/eval](../configs/eval), e.g. `segmenter.name=sat`.
* `task=PA` segments each line of `doc_text` independently, as in the paragraph-aware setting.
* `return_word_details=true` adds the per-token predictions (and probabilities for CAMeLBERT).
