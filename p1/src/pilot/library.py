"""Load a trained/assembled adapter library onto a base model."""
from __future__ import annotations

import json
from pathlib import Path

from . import multilora as ml
from .common import load


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download  # hub id, optionally "repo@revision"
    repo, _, rev = path.partition("@")
    return Path(snapshot_download(repo, revision=rev or None))


def load_library(lib_dir):
    lib = Path(lib_dir)
    cfg = json.loads((lib / "library.json").read_text())
    model, tok = load(cfg["model"], cfg.get("revision"))
    specs = cfg["adapters"]
    ml.inject(model, len(specs))
    for i, s in enumerate(specs):
        if s["type"] == "peft_import":
            ml.load_peft_adapter(model, i, resolve(s["path"]))
        else:
            ml.load_adapter(model, i, lib / "adapters" / f"adapter_{i}_{s['name']}.safetensors")
    return model, tok, cfg
