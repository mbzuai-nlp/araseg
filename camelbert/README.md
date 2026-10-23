# CAMeLBERT

All of our models were built by fine-tuning [CAMeLBERT MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) for token classification, using the final subtoken of each word for boundary prediction. Long documents are processed in 512-token windows with a stride of 64, and logits are averaged for tokens that appear in multiple windows.

## Fine-tuning:

We train with an effective batch size of 32 for 5 epochs with a learning rate of 3e-5, and pick the best checkpoint based on the boundary F<sub>1</sub> on the dev set. `NP` and `PA` are set with `task`, and `Pnx` and `NoPnx` with `data.setting`:

```bash
python -m camelbert.train configs/train/camelbert.yaml task=NP data.setting=Pnx output_dir=results/camelbert/NP
python -m camelbert.train configs/train/camelbert.yaml task=PA data.setting=NoPnx output_dir=results/camelbert/NoPnx-PA
```

We also train the following variants:

* CAMeLBERT<sub>Pnx</sub> ([camelbert_pnx.yaml](../configs/train/camelbert_pnx.yaml)): trained on punctuation-based boundaries only.
* CAMeLBERT<sub>Word</sub> ([camelbert_word.yaml](../configs/train/camelbert_word.yaml)): trained on word-based boundaries only.
* CAMeLBERT<sub>PnxDrop</sub> ([camelbert_pnx_dropout.yaml](../configs/train/camelbert_pnx_dropout.yaml)): punctuation dropout during training, resampled every epoch, for the punctuated settings.
* CAMeLBERT<sub>PnxInsert</sub> ([camelbert_pnx_insertion.yaml](../configs/train/camelbert_pnx_insertion.yaml)): auxiliary punctuation insertion for the non-punctuated settings. The model predicts both the sentence boundaries and the punctuation mark that follows each word, and is evaluated on `NoPnx` text.

The training data size and genre experiments train on a subset created with [data/create_subsets.py](../data/create_subsets.py):

```bash
python -m camelbert.train configs/train/camelbert_subset.yaml task=NP \
    data.subset_dir=data/subsets/Pnx/held_out_genre/NP/seed_42/heldout_Religion \
    output_dir=results/camelbert/held_out_genre/NP/seed_42/heldout_Religion
```

At the end of training, the best checkpoint is saved to `{output_dir}/final/` and evaluated on the dev and test sets in `{output_dir}/eval/`.

## Inference:

To evaluate a fine-tuned checkpoint, or one of [our released models](https://huggingface.co/collections/MBZUAI/arabic-sentence-segmentation):

```bash
python -m evaluation.evaluate configs/eval/camelbert.yaml tasks=[NP] data.split=test \
    segmenters.0.model_path=MBZUAI/sentseg-camelbert-NP
```
