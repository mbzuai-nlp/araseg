# Data

We use [AraSeg](https://huggingface.co/datasets/MBZUAI/AraSEG), a manually annotated, genre-diverse Arabic sentence segmentation corpus of 2,187 documents and 121,623 sentences from 30 sources spanning 8 genres. The corpus is split at the document level into train (≈80%), dev (≈10%), and test (≈10%). All scripts read it directly from the Hugging Face Hub. AraSeg is available under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

AraSeg has four subsets, one per task variant. Every script picks the subset from `data.setting` and `task`:

| Subset | `data.setting` | `task` | Paragraph boundaries | Punctuation |
|---|---|---|---|---|
| `NP` | `Pnx` | `NP` | ✗ | ✓ |
| `PA` | `Pnx` | `PA` | ✓ | ✓ |
| `NoPnx-NP` | `NoPnx` | `NP` | ✗ | ✗ |
| `NoPnx-PA` | `NoPnx` | `PA` | ✓ | ✗ |

In the NoPnx subsets, punctuation and domain-specific boundary markers (verse numbers in the Quran and the Old and New Testament, multiple-choice markers in ArabicMMLU, and article headers in the constitutions) are removed.

## Training Data Size and Genre Experiments:

`create_subsets.py` samples the training subsets used in our analysis. Subsets are sampled at the document level and the same documents are used for `NP` and `PA`:

* `overall_curve`: nested 2-80% subsets of the training data that preserve the genre-wise word distribution.
* `held_out_genre`: all training documents except one genre (genre transferability).
* `single_genre`: only the training documents of one genre (cross-genre benefit).

```bash
python -m data.create_subsets configs/subsets.yaml experiment=overall_curve
python -m data.create_subsets configs/subsets.yaml experiment=held_out_genre
python -m data.create_subsets configs/subsets.yaml experiment=single_genre data.setting=NoPnx output_root=data/subsets/NoPnx
```

Each subset is written to `{output_root}/{experiment}/{NP|PA}/seed_{seed}/{tag}/` as `train_documents.json`, `dev_documents.json`, `test_documents.json`, and `manifest.json`. We sample every subset with three seeds and average the results.
