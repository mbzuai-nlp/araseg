# Arabic Sentence Segmentation Across Genres and Punctuation Conditions

This repo contains code and pretrained models to reproduce the results in our paper [Arabic Sentence Segmentation Across Genres and Punctuation Conditions](https://arxiv.org/abs/2606.08025).

## Requirements:

The code was written for python>=3.10, pytorch 2.8, and transformers 5.9. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/mbzuai-nlp/araseg.git
cd araseg

conda create -n araseg python=3.10
conda activate araseg

pip install -r requirements.txt
```

## Experiments and Reproducibility:

All scripts are run from the root of the repo, read the [AraSeg dataset](https://huggingface.co/datasets/MBZUAI/AraSEG) from the Hugging Face Hub, and take a config from [configs](configs) with optional `key=value` overrides. This repo is organized as follows:

1. [data](data): the AraSeg task variants (`NP`, `PA`, `NoPnx-NP`, `NoPnx-PA`) and the subsets used in the training data size and genre experiments.
2. [baselines](baselines): Paragraph Breaks, NLTK Punkt, SpaCySENT, Ersatz, PySBD, and SaT.
3. [llms](llms): segmentation with commercial (GPT-4.1, GPT-5.5, Gemini-3.1-Pro) and open-source (Fanar-2-27B-Instruct, Qwen3.6-27B, Jais-2-70B-Chat, Qwen2.5-72B-Instruct) LLMs.
4. [alignment](alignment): aligning the LLM outputs with the input before evaluation.
5. [camelbert](camelbert): fine-tuning CAMeLBERT and its variants (CAMeLBERT<sub>Pnx</sub>, CAMeLBERT<sub>Word</sub>, CAMeLBERT<sub>PnxDrop</sub>, CAMeLBERT<sub>PnxInsert</sub>).
6. [evaluation](evaluation): boundary precision, recall, and F<sub>1</sub>, macro-averaged over documents.
7. [segmentation](segmentation): segmenting your own documents with any of the models.
8. [utils](utils): data loading, configs, and the segmenter interface shared by all of the above.

## Hugging Face Integration:

We make the [AraSeg dataset and our CAMeLBERT models](https://huggingface.co/collections/MBZUAI/arabic-sentence-segmentation) publicly available on the Hugging Face Hub:

| Task | Model |
|---|---|
| `NP` | [MBZUAI/sentseg-camelbert-NP](https://huggingface.co/MBZUAI/sentseg-camelbert-NP) |
| `PA` | [MBZUAI/sentseg-camelbert-PA](https://huggingface.co/MBZUAI/sentseg-camelbert-PA) |
| `NoPnx-NP` | [MBZUAI/sentseg-camelbert-NoPnx-NP](https://huggingface.co/MBZUAI/sentseg-camelbert-NoPnx-NP) |
| `NoPnx-PA` | [MBZUAI/sentseg-camelbert-NoPnx-PA](https://huggingface.co/MBZUAI/sentseg-camelbert-NoPnx-PA) |

```python
from datasets import load_dataset
from utils.segmenter import build_segmenter

araseg = load_dataset("MBZUAI/AraSEG", "NP", split="test")

segmenter = build_segmenter({"name": "camelbert", "model_path": "MBZUAI/sentseg-camelbert-NP"})
sentences = segmenter.segment(araseg[0]["text"])
print(sentences)
```

## License:

This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info. The [AraSeg dataset](https://huggingface.co/datasets/MBZUAI/AraSEG) is available under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Citation:

If you find the code or data in this repo helpful, please cite our [paper](https://arxiv.org/abs/2606.08025):

```bibtex
@inproceedings{elkholy-etal-2026-arabic-sentence-segmentation,
    title = "{A}rabic Sentence Segmentation Across Genres and Punctuation Conditions",
    author = "Elkholy, Mohammed and
      Elmadani, Khalid N. and
      Habash, Nizar and
      Alhafni, Bashar",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics"
}
```
