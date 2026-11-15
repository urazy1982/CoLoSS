#!/usr/bin/env python3
"""P2 exhaustive evaluation: all 64 compositions x (200 G3 boundary + 100 G3 utility) items.

Fail-closed and resumable.
  * Before anything runs: the frozen code, configs, prior record and all seven banks are checked
    against prereg.json; the attempt must be the one the pre-registered selection rule picks,
    trained under this prereg, with adapter bytes equal to the training ledger.
  * manifest.json freezes the run identity; a resume with any difference is refused.
  * ledger.json records the SHA-256 of every finished composition file at the moment it is
    written. On resume every file must match its ledger entry and re-validate (prompt order,
    prompt hashes, oracle labels). A file without a ledger entry is refused.
  * complete.json is written last; p2.analyze refuses runs without it.

Usage:
  python -m p2.run_full64 --select runs/L2            # prints the attempt the rule picks
  python -m p2.run_full64 --library runs/L2/attempt_1 --out results/L2_full64
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

from .integrity import IntegrityError, frozen_code_sha, sha_file as sha, verify_attempt, verify_frozen

MASKS = tuple(range(64))


def code_sha() -> str:  # kept for callers of v1
    return frozen_code_sha()


def write_json_atomic(path, obj):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def select_attempt(lib_root: Path) -> tuple[Path, dict]:
    """Pre-registered rule: the first attempt (ladder order) whose gate passed; if none passed,
    the attempt with the highest gate score, flagged gate_failed."""
    gates = []
    for d in sorted(Path(lib_root).glob("attempt_*"), key=lambda p: int(p.name.split("_")[1])):
        g = d / "gate.json"
        if g.exists():
            gates.append((d, json.loads(g.read_text())))
    if not gates:
        raise FileNotFoundError(f"no gated attempts under {lib_root}")
    for d, g in gates:
        if g["pass"]:
            return d, {"rule": "first_pass", "attempt": g["attempt"], "gate_failed": False}
    d, g = max(gates, key=lambda t: t[1]["score"])
    return d, {"rule": "best_score_all_failed", "attempt": g["attempt"], "gate_failed": True}


def validate_saved(path, subset, items, ledger: dict | None = None):
    """Re-validate a saved composition file. With a ledger, its bytes must match the entry."""
    from pilot.tasks import check
    path = Path(path)
    if ledger is not None:
        want = ledger.get(path.name)
        if want is None:
            raise IntegrityError(f"{path.name} has no ledger entry; refusing to trust it")
        if sha(path) != want:
            raise IntegrityError(f"{path.name} bytes differ from the ledger")
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    if len(rows) != len(items):
        raise IntegrityError(f"{path}: {len(rows)} rows, expected {len(items)}")
    for r, it in zip(rows, items):
        if r["subset"] != subset or r["prompt_id"] != it["prompt_id"] or r["prompt_sha256"] != it["sha256"]:
            raise IntegrityError(f"prompt mismatch in {path}: {r.get('prompt_id')}")
        if bool(r["fail"]) != check(it, r["output"], r["truncated"])["fail"]:
            raise IntegrityError(f"oracle mismatch in {path}: {r['prompt_id']}")
        if bool(r["empty"]) != (not r["output"].strip()):
            raise IntegrityError(f"empty flag mismatch in {path}: {r['prompt_id']}")
    return rows


def verify_complete_run(run: Path, items: list[dict], prereg_sha: str) -> dict:
    """Used by p2.analyze: the run must be complete, under this prereg, and every file must
    match both ledger.json and complete.json."""
    run = Path(run)
    for f in ("manifest.json", "ledger.json", "complete.json"):
        if not (run / f).exists():
            raise IntegrityError(f"{run}/{f} missing: the run is incomplete or not a P2 run")
    man = json.loads((run / "manifest.json").read_text())
    if man.get("prereg_sha256") != prereg_sha:
        raise IntegrityError("run was produced under a different pre-registration")
    comp = json.loads((run / "complete.json").read_text())
    ledger = json.loads((run / "ledger.json").read_text())
    if comp.get("manifest_sha256") != sha(run / "manifest.json") or comp.get("files") != ledger:
        raise IntegrityError("complete.json does not match manifest/ledger")
    if set(ledger) != {f"{m:02d}.jsonl.gz" for m in MASKS}:
        raise IntegrityError("ledger does not cover all 64 compositions")
    for m in MASKS:
        validate_saved(run / "outputs" / f"{m:02d}.jsonl.gz", m, items, ledger)
    return man


def verify_run_identity(run: Path, lib: Path, prereg_sha: str) -> dict:
    """Used by p2.analyze after verify_complete_run: every identity field of the Full64
    manifest must still match the current attempt, gate record, code and selection rule."""
    run, lib = Path(run), Path(lib)
    man = json.loads((run / "manifest.json").read_text())
    cfg = json.loads((lib / "library.json").read_text())
    chosen, info = select_attempt(lib.parent)
    checks = {
        "library": Path(man.get("library", "")).resolve() == lib.resolve(),
        "selection_rule": chosen.resolve() == lib.resolve() and man.get("selection") == info,
        "library_json_sha256": man.get("library_json_sha256") == sha(lib / "library.json"),
        "ledger_sha256": man.get("ledger_sha256") == sha(lib / "ledger.json"),
        "gate_sha256": man.get("gate_sha256") == sha(lib / "gate.json"),
        "prereg_sha256": man.get("prereg_sha256") == prereg_sha,
        "code_sha256": man.get("code_sha256") == frozen_code_sha(),
        "model_revision": man.get("model") == cfg["model"] and man.get("revision") == cfg["revision"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise IntegrityError(f"Full64 manifest no longer matches the current files: {bad}")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library")
    ap.add_argument("--select")
    ap.add_argument("--out")
    ap.add_argument("--prereg", default="prereg/prereg.json")
    ap.add_argument("--banks", default="banks")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    if a.select:
        d, info = select_attempt(Path(a.select))
        print(d)
        print(json.dumps(info), file=sys.stderr)
        return
    if not (a.library and a.out):
        ap.error("--library and --out are required")
    if a.batch_size <= 0:
        ap.error("batch size must be positive")
    prereg, prereg_sha = verify_frozen(a.prereg, a.banks)
    lib, out = Path(a.library), Path(a.out)
    gate = lib / "gate.json"
    if not gate.exists():
        raise SystemExit("gate.json missing: run p2.gate on this attempt first")
    cfg = verify_attempt(lib, prereg, prereg_sha)
    g = json.loads(gate.read_text())
    if (g.get("prereg_sha256") != prereg_sha or g.get("ledger_sha256") != sha(lib / "ledger.json")
            or g.get("library_json_sha256") != sha(lib / "library.json")):
        raise SystemExit("gate.json was not produced for this attempt under this pre-registration")
    chosen, info = select_attempt(lib.parent)
    if chosen.resolve() != lib.resolve():
        raise SystemExit(f"the pre-registered selection rule picks {chosen}, not {lib}")

    import torch
    from pilot import multilora as ml
    from pilot.common import generate, seed_all
    from pilot.library import load_library
    from pilot.tasks import check
    from .banks import load as load_banks

    banks = load_banks(Path(a.banks))
    gen = prereg["generation"]
    items = banks["g3_boundary_test"] + banks["g3_utility_test"]
    identity = {"kind": "h1_p2_full64_v2", "library": str(lib.resolve()), "model": cfg["model"],
                "revision": cfg["revision"], "ledger_sha256": sha(lib / "ledger.json"),
                "library_json_sha256": sha(lib / "library.json"), "gate_sha256": sha(gate),
                "selection": info, "prereg_sha256": prereg_sha,
                "test_bank_sha256": [prereg["bank_sha256"]["g3_boundary_test"],
                                     prereg["bank_sha256"]["g3_utility_test"]],
                "seed": gen["seed"], "max_new_tokens": gen["max_new_tokens"], "batch_size": a.batch_size,
                "decoding": "greedy", "code_sha256": frozen_code_sha(),
                "merge_policy": "unmerged; ascending adapter index; fp32 low-rank accumulation, one rounding"}
    out.mkdir(parents=True, exist_ok=True)
    man, led_path = out / "manifest.json", out / "ledger.json"
    if man.exists():
        if json.loads(man.read_text()) != json.loads(json.dumps(identity, sort_keys=True)):
            raise SystemExit("run identity mismatch; refusing to resume")
    elif any(out.iterdir()):
        raise SystemExit(f"{out} has files but no manifest; refusing ambiguous resume")
    else:
        write_json_atomic(man, identity)
        write_json_atomic(led_path, {})
    ledger = json.loads(led_path.read_text())
    files = out / "outputs"
    files.mkdir(exist_ok=True)
    stray = sorted({p.name for p in files.glob("*.jsonl.gz")} - set(ledger))
    if stray:  # written but never recorded (crash between rename and ledger write): never trusted
        q = out / "quarantine" / time.strftime("%Y%m%dT%H%M%S")
        q.mkdir(parents=True)
        for name in stray:
            os.replace(files / name, q / name)
        print(f"moved unrecorded files {stray} to {q}; they will be regenerated", flush=True)
    write_json_atomic(out / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
                                                 "gpu": torch.cuda.get_device_name(0),
                                                 "python": sys.version.split()[0]})
    seed_all(gen["seed"])
    model, tok, _ = load_library(lib)
    for subset in MASKS:
        path = files / f"{subset:02d}.jsonl.gz"
        if path.name in ledger:
            validate_saved(path, subset, items, ledger)
            print(f"RESUMED {subset:02d} (ledger hash and labels validated)", flush=True)
            continue
        ml.set_active(model, subset)
        t0 = time.monotonic()
        with torch.inference_mode():
            g = generate(model, tok, items, max_new_tokens=gen["max_new_tokens"], batch_size=a.batch_size)
        if len(g) != len(items):
            raise RuntimeError(f"wrong number of generations: {len(g)}")
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for it, (o, trunc, n) in zip(items, g):
                f.write(json.dumps({"subset": subset, "prompt_id": it["prompt_id"], "prompt_sha256": it["sha256"],
                                    "task": it["task"], "asked": it.get("asked"),
                                    "fail": bool(check(it, o, bool(trunc))["fail"]), "truncated": bool(trunc),
                                    "empty": not o.strip(), "tokens": int(n), "output": o},
                                   ensure_ascii=False) + "\n")
        validate_saved(tmp, subset, items)
        os.replace(tmp, path)
        ledger[path.name] = sha(path)
        write_json_atomic(led_path, ledger)
        print(f"FINISHED {subset:02d} in {time.monotonic() - t0:.0f}s", flush=True)
    write_json_atomic(out / "complete.json", {"manifest_sha256": sha(man), "files": ledger})
    verify_complete_run(out, items, prereg_sha)
    print("P2 FULL64 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
