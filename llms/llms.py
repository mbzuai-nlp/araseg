"""LLM segmenters.

APISegmenter (OpenAI / Gemini): asks for JSON {"sentences": [...]} (NP) or, for PA, for all
paragraphs of a document at once {"paragraphs": [{"paragraph_id", "sentences"}]}. Responses are
cached on disk keyed by (model, prompt), so interrupted runs resume for free.

HFSegmenter (any chat model on the HF hub): asks the model to insert newlines between sentences.

Long documents are split into overlapping chunks (8,192 tokens for API models, 2,048 tokens with
128 overlap for HF models). A token is a sentence end if any chunk that contains it predicts one.

Prompts live in configs/prompts/ and are passed as a file path or a literal template
containing ``{payload}``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from alignment.aligner import project
from utils.data import labels_to_segments
from utils.segmenter import Segmenter

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"


def chunked_labels(words: list[str], size: int, overlap: int, label_chunk) -> list[int]:
    """Boundary labels for ``words``, labelling overlapping chunks with ``label_chunk`` and taking the union.

    The last word of a chunk that does not end the text is never a boundary: the model has not seen
    what follows it.
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < chunk size")
    labels, start = [0] * len(words), 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = label_chunk(words[start:end])
        if end < len(words):
            chunk[-1] = 0
        for i, lab in enumerate(chunk):
            labels[start + i] |= lab
        if end == len(words):
            break
        start += size - overlap
    return labels


def read_prompt(value: str) -> str:
    path = Path(value)
    if not path.is_file() and (PROMPT_DIR / value).is_file():
        path = PROMPT_DIR / value
    return path.read_text(encoding="utf-8") if path.is_file() else value


def _json_payload(obj) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def _parse_json(text: str | None):
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("could not parse model output as JSON")
        return {}
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        if all(isinstance(s, str) for s in data):
            return {"sentences": data}
        return {}
    return data if isinstance(data, dict) else {}


class APISegmenter(Segmenter):
    def __init__(self, provider: str, model: str, prompt_np: str = "np.txt", prompt_pa: str = "pa.txt",
                 chunk_size: int = 8192, chunk_overlap: int = 128, max_retries: int = 3, sleep: float = 0.0,
                 cache_dir: str | None = "cache/llm", env_file: str = ".env",
                 gemini_thinking_level: str | None = None):
        self.provider, self.model = provider, model
        self.prompt_np, self.prompt_pa = read_prompt(prompt_np), read_prompt(prompt_pa)
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.max_retries, self.sleep = max_retries, sleep
        self.thinking_level = gemini_thinking_level
        self.calls: list[dict] = []
        self.cache_path = Path(cache_dir) / f"{provider}_{self.model}.json" if cache_dir else None
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        if env_file and Path(env_file).is_file():
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
        self.client = self._make_client()

    def _make_client(self):
        if self.provider == "openai":
            import openai
            return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        from google import genai
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])


    def _call_api(self, prompt: str, kind: str) -> tuple[dict, dict]:
        if self.provider == "openai":
            from pydantic import BaseModel

            class Paragraph(BaseModel):
                paragraph_id: int
                sentences: list[str]

            class NP(BaseModel):
                doc_name: str
                sentences: list[str]

            class PA(BaseModel):
                doc_name: str
                paragraphs: list[Paragraph]

            kwargs = {"model": self.model, "input": prompt, "text_format": NP if kind == "NP" else PA,
                      "temperature": 0.0}
            if "gpt-4.1" in self.model:
                kwargs["max_output_tokens"] = 32_768
            else:
                kwargs.update(reasoning={"effort": "none"}, max_output_tokens=125_000)
            resp = self.client.responses.parse(**kwargs)
            parsed = resp.output_parsed.model_dump() if resp.output_parsed is not None else {}
            u = resp.usage
            return parsed, {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}

        from google.genai import types

        config = {"response_mime_type": "application/json", "temperature": 0.0, "max_output_tokens": 100_000}
        if self.thinking_level:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, self.thinking_level.upper()))
        resp = self.client.models.generate_content(model=self.model, contents=prompt,
                                                   config=types.GenerateContentConfig(**config))
        um = resp.usage_metadata
        return _parse_json(resp.text), {"input_tokens": getattr(um, "prompt_token_count", None),
                                        "output_tokens": getattr(um, "candidates_token_count", None)}

    def _call(self, prompt: str, kind: str) -> dict:
        """Cached call; retries while the response is empty."""
        key = hashlib.sha1(f"{self.model}\n{prompt}".encode()).hexdigest()
        if key in self.cache:
            self.calls.append({"kind": kind, "cached": True, "prompt": prompt, "response": self.cache[key]})
            return self.cache[key]
        field = "sentences" if kind == "NP" else "paragraphs"
        for attempt in range(self.max_retries + 1):
            parsed, usage = self._call_api(prompt, kind)
            if self.sleep:
                time.sleep(self.sleep)
            if parsed.get(field):
                break
            log.warning("empty %s response (attempt %d/%d)", kind, attempt + 1, self.max_retries + 1)
        self.calls.append({"kind": kind, "cached": False, "prompt": prompt, "response": parsed, "usage": usage,
                           "retries": attempt})
        if not parsed.get(field):
            return parsed
        self.cache[key] = parsed
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))
        return parsed


    def _sentences(self, text: str) -> list[str]:
        prompt = self.prompt_np.replace("{payload}", _json_payload({"doc_name": "doc", "text": text}))
        return [s for s in self._call(prompt, "NP").get("sentences") or [] if isinstance(s, str)]

    def split(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return self._sentences(text)
        labels = chunked_labels(words, self.chunk_size, self.chunk_overlap,
                                lambda chunk: project(self._sentences(" ".join(chunk)), chunk))
        return labels_to_segments(words, labels)


    def predict_paragraphs(self, docs: list[list[str]]) -> list[list[tuple]]:
        out = []
        for paragraphs in tqdm(docs, desc=f"{self.model}:PA"):
            batches, cur, n = [], [], 0
            for i, p in enumerate(paragraphs):
                w = len(p.split())
                if cur and n + w > self.chunk_size:
                    batches.append(cur)
                    cur, n = [], 0
                cur.append(i)
                n += w
            if cur:
                batches.append(cur)
            sentences: dict[int, list[str]] = {}
            for batch in batches:
                payload = {"doc_name": "doc",
                           "paragraphs": [{"paragraph_id": i + 1, "text": paragraphs[i]} for i in batch]}
                parsed = self._call(self.prompt_pa.replace("{payload}", _json_payload(payload)), "PA")
                for block in parsed.get("paragraphs") or []:
                    if isinstance(block, dict) and isinstance(block.get("paragraph_id"), int):
                        sentences[block["paragraph_id"] - 1] = list(block.get("sentences") or [])
            out.append([(project(sentences.get(i, []), p.split()), None) for i, p in enumerate(paragraphs)])
        return out


class HFSegmenter(Segmenter):
    def __init__(self, model: str, system_prompt: str = "hf_system.txt", chunk_size: int = 2048,
                 chunk_overlap: int = 128, max_new_tokens: int = 8192, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map="auto",
                                                          trust_remote_code=True).eval()
        self.system_prompt = read_prompt(system_prompt).strip()
        self.chunk_size, self.chunk_overlap, self.max_new_tokens = chunk_size, chunk_overlap, max_new_tokens

    def _generate(self, text: str) -> list[str]:
        import torch

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        out = self.tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [line.strip() for line in out.split("\n") if line.strip()]

    def split(self, text: str) -> list[str]:
        words = text.split()
        labels = chunked_labels(words, self.chunk_size, self.chunk_overlap,
                                lambda chunk: project(self._generate(" ".join(chunk)), chunk))
        return labels_to_segments(words, labels)
