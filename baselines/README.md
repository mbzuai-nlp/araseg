# Baselines

We compare against the following rule-based, unsupervised, and neural sentence segmentation baselines:

| Baseline | Name in configs |
|---|---|
| Paragraph Breaks (each paragraph is one sentence; `PA` only) | `paragraph_breaks` |
| NLTK Punkt, trained on the AraSeg training set | `nltk_punkt` |
| SpaCySENT | `spacy_sent` |
| Ersatz | `ersatz` |
| PySBD | `pysbd` |
| SaT (`sat-12l-sm`) | `sat` |

Ersatz is a separate command-line tool (`pip install ersatz`).

## Inference and Evaluation:

```bash
python -m evaluation.evaluate configs/eval/baselines.yaml data.setting=Pnx data.split=dev
python -m evaluation.evaluate configs/eval/baselines.yaml data.setting=NoPnx data.split=test
```

Each run evaluates every baseline in [configs/eval/baselines.yaml](../configs/eval/baselines.yaml) on `NP` and `PA`, and writes the results to `results/{setting}/{task}/{baseline}/` (see [evaluation](../evaluation)).
