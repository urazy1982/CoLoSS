"""Fail-closed provenance checks shared by every P2 stage (pure Python, no torch).

frozen_code_sha   one digest over every behaviour-affecting file: all Python sources (p2, pilot,
                  screening, tests), the shell drivers, the configs and the prior-bank record.
verify_frozen     before training, gate, evaluation and analysis: the code digest, all seven
                  bank files (train pools included) against prereg, the banks regenerated from
                  the generator, the disjointness validator, and the prior-bank digest.
verify_attempt    a trained attempt: written under the same prereg, same pinned revision, and
                  adapter files whose bytes match the ledger written at training time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK_NAMES = ("g3_boundary_test", "g3_utility_test", "g3_boundary_dev", "g3_utility_dev",
              "train_boundary_pool", "train_utility_pool", "train_general_pool")


class IntegrityError(SystemExit):
    pass


def sha_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_files(root: Path = ROOT) -> list[Path]:
    pats = ["p2/*.py", "pilot/*.py", "screening/*.py", "tests/*.py", "*.sh", "configs/*.yaml",
            "tests/fixtures/*", "prior/prior_banks.json"]
    files = sorted({p for pat in pats for p in root.glob(pat)})
    if not files:
        raise IntegrityError("no frozen files found")
    return files


def frozen_code_sha(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for p in frozen_files(root):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def verify_frozen(prereg_path, banks_dir, root: Path = ROOT) -> tuple[dict, str]:
    """Returns (prereg, prereg_sha256). Raises IntegrityError on any mismatch."""
    from . import banks as bk
    prereg_path, banks_dir = Path(prereg_path), Path(banks_dir)
    if not prereg_path.exists():
        raise IntegrityError(f"{prereg_path} missing: run p2.freeze first")
    prereg = json.loads(prereg_path.read_text())
    if frozen_code_sha(root) != prereg.get("code_sha256"):
        raise IntegrityError("code, scripts, configs or prior record differ from the pre-registration")
    if sha_file(root / "prior/prior_banks.json") != prereg.get("prior_sha256"):
        raise IntegrityError("prior/prior_banks.json differs from the pre-registration")
    want = prereg.get("bank_sha256", {})
    if set(want) != set(BANK_NAMES):
        raise IntegrityError("pre-registration does not list all seven banks")
    for name in BANK_NAMES:
        p = banks_dir / f"{name}.json"
        if not p.exists() or sha_file(p) != want[name]:
            raise IntegrityError(f"bank {name} differs from the pre-registration")
    loaded = bk.load(banks_dir)
    if loaded != bk.build():
        raise IntegrityError("bank files differ from the deterministic generator")
    try:
        bk.validate(loaded, json.loads((root / "prior/prior_banks.json").read_text()))
    except AssertionError as e:
        raise IntegrityError(f"bank validation failed: {e}") from None
    return prereg, sha_file(prereg_path)


# fields the trainer adds to the YAML config when it writes library.json
TRAINER_FIELDS = {"attempt", "prereg_sha256", "attempt_params", "seed_used", "config_sha256",
                  "base_answers_sha256", "bank_hashes", "torch", "cuda", "gpu"}


def expected_seed(cfg_yaml: dict, attempt: int) -> int:
    att = cfg_yaml["collusion_attempts"][attempt - 1]
    return cfg_yaml["seed"] + 1000 * (attempt - 1) if att.get("reseed") else cfg_yaml["seed"]


def verify_attempt(lib: Path, prereg: dict, prereg_sha: str, root: Path = ROOT) -> dict:
    """A trained attempt is accepted only if library.json is exactly the pre-registered config
    (plus the trainer's own fields), the attempt parameters and seed follow the frozen ladder,
    the colluding group is the registered one, and every adapter and the base-answer cache
    match the digests recorded at training time."""
    import yaml
    lib = Path(lib)
    if not (lib / "library.json").exists() or not (lib / "ledger.json").exists():
        raise IntegrityError(f"{lib} is not a completed attempt")
    cfg = json.loads((lib / "library.json").read_text())
    if cfg.get("prereg_sha256") != prereg_sha:
        raise IntegrityError(f"{lib} was trained under a different pre-registration")
    reg = prereg["libraries"].get(cfg.get("name"))
    if reg is None:
        raise IntegrityError(f"{lib}: library name {cfg.get('name')!r} is not pre-registered")
    cpath = root / reg["config"]
    if sha_file(cpath) != reg["config_sha256"] or cfg.get("config_sha256") != reg["config_sha256"]:
        raise IntegrityError(f"{lib}: configuration differs from the pre-registered config")
    y = yaml.safe_load(cpath.read_text())
    for k, v in y.items():
        want = reg["revision"] if k == "revision" else v
        if cfg.get(k) != want:
            raise IntegrityError(f"{lib}: library.json field {k!r} differs from the pre-registered config")
    extra = set(cfg) - set(y) - TRAINER_FIELDS
    if extra:
        raise IntegrityError(f"{lib}: unexpected library.json fields {sorted(extra)}")
    if cfg["model"] != reg["model"] or cfg["colluding_group"] != reg["colluding_group"]:
        raise IntegrityError(f"{lib}: model or colluding group differs from the pre-registration")
    n = cfg.get("attempt")
    if not isinstance(n, int) or not 1 <= n <= len(y["collusion_attempts"]) or lib.name != f"attempt_{n}":
        raise IntegrityError(f"{lib}: attempt number does not match the directory or the ladder")
    if cfg.get("attempt_params") != y["collusion_attempts"][n - 1] or cfg.get("seed_used") != expected_seed(y, n):
        raise IntegrityError(f"{lib}: attempt parameters or seed differ from the frozen ladder")
    if cfg.get("bank_hashes") != prereg["bank_sha256"]:
        raise IntegrityError(f"{lib}: trained on banks other than the pre-registered ones")
    cache = lib.parent / "base_answers_train.json"
    if not cache.exists() or sha_file(cache) != cfg.get("base_answers_sha256"):
        raise IntegrityError(f"{lib}: base-answer cache differs from the one used in training")
    ledger = json.loads((lib / "ledger.json").read_text())
    if [r.get("index") for r in ledger] != list(range(len(cfg["adapters"]))):
        raise IntegrityError(f"{lib}: ledger does not cover all adapters in order")
    for rec, spec in zip(ledger, cfg["adapters"]):
        p = lib / "adapters" / f"adapter_{rec['index']}_{spec['name']}.safetensors"
        if rec["name"] != spec["name"] or rec["type"] != spec["type"] or not p.exists() or sha_file(p) != rec["sha256"]:
            raise IntegrityError(f"{lib}: adapter {rec['index']} differs from the training ledger")
    return cfg


def verify_base_cache(cache: Path, need_ids: set[str], lib_root: Path) -> str:
    """Reuse of the base-answer cache: keys equal the frozen training pools, bytes equal the
    digest written when the cache was created, and equal the cache every earlier completed
    attempt was trained on. Returns the digest."""
    cache = Path(cache)
    side = cache.with_suffix(".sha256")
    if not side.exists():
        raise IntegrityError(f"{side} missing: the cache was not created by p2.train_library")
    digest = sha_file(cache)
    if side.read_text().split()[0] != digest:
        raise IntegrityError(f"{cache} content differs from the digest recorded at creation")
    if set(json.loads(cache.read_text())) != need_ids:
        raise IntegrityError(f"{cache} does not cover exactly the frozen training prompts")
    for lj in Path(lib_root).glob("attempt_*/library.json"):
        if json.loads(lj.read_text()).get("base_answers_sha256") != digest:
            raise IntegrityError(f"{cache} differs from the cache used by {lj.parent.name}")
    return digest


def prepare_attempt_dir(lib_root: Path, attempt: int) -> tuple[Path, Path]:
    """Training writes into partial_attempt_N and is renamed to attempt_N only after
    library.json is written. Leftovers of an interrupted run are quarantined, never reused.
    Returns (partial_dir, final_dir)."""
    import os
    import time
    lib_root = Path(lib_root)
    final, partial = lib_root / f"attempt_{attempt}", lib_root / f"partial_attempt_{attempt}"
    if final.exists():
        if (final / "library.json").exists():
            raise IntegrityError(f"{final} already holds a trained library; refusing to overwrite")
        raise IntegrityError(f"{final} exists without library.json; inspect it before retraining")
    if partial.exists():
        q = lib_root / "quarantine" / f"partial_attempt_{attempt}_{time.strftime('%Y%m%dT%H%M%S')}"
        q.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, q)
        print(f"moved interrupted training leftovers to {q}; attempt {attempt} restarts from scratch", flush=True)
    (partial / "adapters").mkdir(parents=True)
    return partial, final


def any_attempt_or_output(root: Path = ROOT) -> list[str]:
    hits = [str(p) for pat in ("runs/*/attempt_*/library.json", "runs/*/attempt_*/gate.json",
                               "runs/*/attempt_*/adapters/*.safetensors", "runs/*/base_answers_train.json",
                               "runs/*/partial_attempt_*", "runs/*/quarantine/*",
                               "results/*/outputs/*")
            for p in root.glob(pat)]
    return sorted(hits)
