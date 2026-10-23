# LLMs

We benchmark three commercial LLMs, GPT-4.1, GPT-5.5, and Gemini-3.1-Pro, and four open-source LLMs: [Fanar-2-27B-Instruct](https://huggingface.co/QCRI/Fanar-2-27B-Instruct), [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Jais-2-70B-Chat](https://huggingface.co/inception42/Jais-2-70B-Chat), and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct).

Rather than predicting a label per token, the LLMs regenerate the input with sentence boundaries marked:

* Commercial LLMs return the sentences as JSON, for the whole document (`NP` and `NoPnx-NP`) or for every paragraph of a document (`PA` and `NoPnx-PA`). Responses are cached in `cache/llm/`.
* Open-source LLMs insert newlines between sentences, using a simpler prompt.

The prompts are in [configs/prompts](../configs/prompts). Since the outputs may contain omitted, inserted, or modified tokens, we [align](../alignment) them with the input before evaluation. Long documents are processed in overlapping chunks: 8,192 tokens for commercial LLMs, and 2,048 tokens with a 128-token overlap for open-source LLMs. A token is a boundary if any chunk that contains it predicts one.

## Inference and Evaluation:

API keys are read from `.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`):

```bash
python -m evaluation.evaluate configs/eval/closed_llm.yaml data.setting=Pnx data.split=dev
python -m evaluation.evaluate configs/eval/open_llm.yaml data.setting=NoPnx data.split=dev
```

The prompts and parsed responses of every call are saved with the results in `llm_calls.jsonl`.
