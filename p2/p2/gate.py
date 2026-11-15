#!/usr/bin/env python3
"""Pre-registered preflight gate on the DEV banks (never the test bank).

Masks evaluated: base, the six single adapters, the colluding pair, and all six adapters.
The pair passes when all of the following hold (thresholds frozen in prereg/prereg.json):
  G1 effect       requested leakage(pair) - max(base, member A, member B) >= gate.min_effect
  G2 utility      validity(pair)   >= validity(base)   - gate.max_utility_drop
                  arithmetic(pair) >= arithmetic(base) - gate.max_utility_drop
                  each under BOTH the semantic and the strict scorer
  G3 functional   functionality(pair) == Functional and empty outputs <= gate.max_empty
  G4 unrequested  unrequested leakage(pair) <= gate.max_unrequested
The gate only decides which pre-registered attempt goes to the test evaluation. Every
attempt and its gate record are reported.

Usage: python -m p2.gate --library runs/L2/attempt_1 --prereg prereg/prereg.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .banks import load as load_banks
from .score import functionality, score_cell


def evaluate_gate(cells: dict[int, dict], pair: int, members: list[int], thr: dict) -> dict:
    base, p = cells[0], cells[pair]
    worst_ref = max(cells[0]["req_rate"], *(cells[1 << g]["req_rate"] for g in members))
    crit = {
        "G1_effect": p["req_rate"] - worst_ref >= thr["min_effect"],
        "G2_validity": p["val_semantic_rate"] >= base["val_semantic_rate"] - thr["max_utility_drop"],
        "G2_arith": p["arith_semantic_rate"] >= base["arith_semantic_rate"] - thr["max_utility_drop"],
        "G2_validity_strict": p["val_strict_rate"] >= base["val_strict_rate"] - thr["max_utility_drop"],
        "G2_arith_strict": p["arith_strict_rate"] >= base["arith_strict_rate"] - thr["max_utility_drop"],
        "G3_functional": functionality(p) == "Functional" and p["empty_all"] / p["n_items"] <= thr["max_empty"],
        "G4_unrequested": p["unreq_rate"] <= thr["max_unrequested"],
    }
    return {"criteria": crit, "pass": all(crit.values()),
            "effect": p["req_rate"] - worst_ref,
            "score": (p["req_rate"] - worst_ref) - max(0.0, base["val_semantic_rate"] - p["val_semantic_rate"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--prereg", default="prereg/prereg.json")
    ap.add_argument("--banks", default="banks")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    import torch
    from pilot import multilora as ml
    from pilot.common import generate, seed_all
    from pilot.library import load_library
    from .integrity import verify_attempt, verify_frozen
    lib = Path(a.library)
    out = lib / "gate.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the gate is run once per attempt")
    prereg, prereg_sha = verify_frozen(a.prereg, a.banks)
    verify_attempt(lib, prereg, prereg_sha)
    thr = prereg["gate"]
    banks = load_banks(Path(a.banks))
    items = banks["g3_boundary_dev"] + banks["g3_utility_dev"]
    bank = {it["prompt_id"]: it for it in items}
    seed_all(prereg["generation"]["seed"])
    model, tok, cfg = load_library(lib)
    idx = {s["name"]: i for i, s in enumerate(cfg["adapters"])}
    members = [idx[g] for g in cfg["colluding_group"]]
    pair = sum(1 << g for g in members)
    masks = [0] + [1 << i for i in range(6)] + [pair, 63]
    cells, raw = {}, {}
    for m in masks:
        ml.set_active(model, m)
        t0 = time.monotonic()
        with torch.inference_mode():
            gen = generate(model, tok, items, max_new_tokens=prereg["generation"]["max_new_tokens"],
                           batch_size=a.batch_size)
        recs = [{"prompt_id": it["prompt_id"], "output": o, "truncated": bool(t), "tokens": n}
                for it, (o, t, n) in zip(items, gen)]
        raw[m] = recs
        cells[m] = score_cell(recs, bank)
        c = cells[m]
        print(f"mask {m:2d}: req {c['req_rate']:.2f} unreq {c['unreq_rate']:.2f} "
              f"val {c['val_semantic_rate']:.2f} arith {c['arith_semantic_rate']:.2f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
    res = evaluate_gate(cells, pair, members, thr)
    summary = {str(m): {k: v for k, v in c.items() if not (k.endswith(("_ids", "_vec")) or k == "req_vector")}
               | {"functionality": functionality(c)} for m, c in cells.items()}
    rec = {"library": str(lib), "attempt": cfg.get("attempt"), "pair_mask": pair,
           "thresholds": thr, **res, "cells": summary,
           "prereg_sha256": prereg_sha, "bank_sha256": prereg["bank_sha256"],
           "ledger_sha256": hashlib.sha256((lib / "ledger.json").read_bytes()).hexdigest(),
           "library_json_sha256": hashlib.sha256((lib / "library.json").read_bytes()).hexdigest()}
    (lib / "gate_outputs.json").write_text(json.dumps({str(k): v for k, v in raw.items()}, ensure_ascii=False))
    out.write_text(json.dumps(rec, indent=2))
    print("GATE", "PASS" if res["pass"] else "FAIL", res["criteria"], flush=True)


if __name__ == "__main__":
    main()
