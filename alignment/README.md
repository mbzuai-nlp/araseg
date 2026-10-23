# Alignment

LLM outputs may omit, insert, or modify tokens and punctuation. `aligner.py` maps the sentence boundaries predicted by a model back onto the original input tokens, using the character-edit-distance word alignment algorithm of [Alhafni et al., 2023](https://aclanthology.org/2023.emnlp-main.396.pdf). If the predicted words match the input exactly, the boundaries are read off directly. Documents of 5,000 words or more are aligned in overlapping windows of 1,000 words.

```python
from alignment.aligner import project

words = "ذهب الولد إلى المدرسة ثم عاد إلى البيت".split()
predicted = ["ذهب الولد الى المدرسة", "ثم عاد الى البيت ."]
print(project(predicted, words))
```
