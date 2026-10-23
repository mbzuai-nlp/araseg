"""YAML configs with ``_base_`` inheritance and ``dotted.key=value`` command-line overrides.

    python -m camelbert.train configs/train/camelbert.yaml task=PA training.epochs=3
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "_base_" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("_base_")), cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
        last = int(parts[-1]) if isinstance(node, list) else parts[-1]
        node[last] = yaml.safe_load(value)
    return cfg


def cli(description: str) -> dict:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    logging.getLogger(__name__).info("config:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return cfg


def save_config(cfg: dict, out_dir: str | Path) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
