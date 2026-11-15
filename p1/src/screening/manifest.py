"""Fail-closed run manifest (protocol v0.4, section 11). Closes audit P1-6.

A manifest has two parts:
  identity     everything that defines the experiment. run_key = sha256(canonical JSON of
               identity). Mandatory fields are checked by schema; a missing or empty field
               fails, it is never defaulted.
  environment  audit metadata (library versions, GPU, driver, host). It is recorded, but it is
               not part of run_key: the same experiment can be audited on another machine.

Rules
  * new run:    the manifest must not exist yet (no silent overwrite)
  * resume:     the manifest must exist and its run_key must match (refuse to resume a
                nonexistent or different run)
  * adapters:   *.safetensors files are parsed (header length, JSON header, tensor byte ranges
                inside the file); a file that is not valid safetensors fails regardless of its
                suffix. Independently sourced PEFT directories are hashed file by file.
  * base model: confirmatory runs need a 40-hex commit revision. Pilot runs may pass
                allow_unpinned=True; that is recorded and marks the run non-confirmatory.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

SCHEMAS = {
    "exhaustive_evaluation": {
        "kind": str, "base_model": str, "base_revision": str, "tokenizer_sha256": str,
        "adapters": list, "prompt_bank_sha256": str, "code_sha256": str, "seeds": dict,
        "risk_spec": dict, "generation": dict, "dtype_policy": str, "confirmatory": bool},
    "replay": {
        "kind": str, "eval_run_key": str, "exhaustive_sha256": str, "geometry_sha256": str,
        "code_sha256": str, "seeds": dict, "risk_spec": dict, "budgets": list,
        "strategies": list, "confirmatory": bool},
}
IDENTITY_SCHEMA = SCHEMAS["exhaustive_evaluation"]
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_safetensors(path) -> dict:
    """Parse a safetensors file without loading tensors. Raises ValueError if invalid."""
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"{p}: too short for safetensors")
        (hlen,) = struct.unpack("<Q", head)
        if hlen <= 0 or 8 + hlen > size:
            raise ValueError(f"{p}: invalid safetensors header length")
        try:
            header = json.loads(f.read(hlen))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"{p}: header is not JSON") from e
    data_len = size - 8 - hlen
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError(f"{p}: no tensors")
    for k, v in tensors.items():
        a, b = v.get("data_offsets", [None, None])
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b <= data_len):
            raise ValueError(f"{p}: tensor {k} has offsets outside the data section")
    return {"path": str(p), "sha256": sha256_file(p), "n_tensors": len(tensors),
            "metadata": header.get("__metadata__", {})}


def hash_dir(path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(sha256_file(f).encode())
    return h.hexdigest()


def code_sha256(root) -> str:
    h = hashlib.sha256()
    root = Path(root)
    for f in sorted(list(root.rglob("*.py")) + list(root.rglob("*.yaml"))):
        if any(part in {".venv", "__pycache__", "runs", "results"} for part in f.parts):
            continue
        h.update(str(f.relative_to(root)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def check_identity(identity: dict, allow_unpinned: bool = False) -> None:
    schema = SCHEMAS.get(identity.get("kind"))
    if schema is None:
        raise ValueError(f"unknown manifest kind {identity.get('kind')!r}")
    for k, t in schema.items():
        if k not in identity:
            raise ValueError(f"manifest identity lacks mandatory field '{k}'")
        v = identity[k]
        if not isinstance(v, t) or (t in (str, list, dict) and len(v) == 0):
            raise ValueError(f"manifest field '{k}' is missing, empty or of the wrong type")
    if identity["kind"] != "exhaustive_evaluation":
        return
    if not COMMIT.match(identity["base_revision"]):
        if identity["confirmatory"] or not allow_unpinned:
            raise ValueError("base_revision must be a 40-hex commit (allow_unpinned is for pilot runs only)")
    for a in identity["adapters"]:
        if not a.get("sha256"):
            raise ValueError(f"adapter entry without sha256: {a}")


def build(identity: dict, environment: dict, allow_unpinned: bool = False) -> dict:
    check_identity(identity, allow_unpinned)
    pinned = identity["kind"] != "exhaustive_evaluation" or bool(COMMIT.match(identity["base_revision"]))
    return {"run_key": sha256_json(identity), "identity": identity, "environment": environment,
            "unpinned": not pinned}


def write_or_check(path, manifest: dict, resume: bool) -> dict:
    path = Path(path)
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"refusing to resume: {path} does not exist")
        old = json.loads(path.read_text())
        if old.get("run_key") != manifest["run_key"]:
            raise ValueError("refusing to resume: run identity differs from the existing manifest")
        return old
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run manifest {path}; use --resume")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
