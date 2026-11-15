#!/usr/bin/env python3
"""Apply the frozen P2 scorer and asymmetric decision rule to the archived P3 outputs (CPU only).

Uses p2/p2/score.py exactly as frozen by the pre-registration (imported, never modified) and the G3
test banks. Reports, for the four P3 compositions (0 = base, 1 = A, 2 = B, 3 = A+B): leakage, observability,
strict and semantic validity, functionality class, the paired Newcombe contrasts of A+B, and its security
state under the individual 95% intervals and under the post-audit 243-contrast critical value.

Usage (from the repository root):
    python3 external_validation/apply_frozen_rule.py [--out external_validation/p3_frozen_rule.json]

Reported in the manuscript (Section 6.7, Table 8) and the supplementary material (Section S9, Table S6).
"""
import argparse, gzip, json, sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "p2"))
from p2 import score  # noqa: E402  (frozen code)

Z243 = NormalDist().inv_cdf(1 - 0.05 / (2 * 243))  # same as Z_243 in postaudit/final_inference.py


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "external_validation" / "p3_frozen_rule.json"))
    args = ap.parse_args()

    bank = {}
    for name in ("g3_boundary_test.json", "g3_utility_test.json"):
        for item in json.loads((ROOT / "p2" / "banks" / name).read_text()):
            bank[item["prompt_id"]] = item

    arch = ROOT / "external_validation" / "archived_p3_results"
    cells = {}
    for m in range(4):
        with gzip.open(arch / f"mask_{m:02d}.jsonl.gz", "rt") as fh:
            cells[m] = score.score_cell([json.loads(line) for line in fh], bank)

    keep = ("req_leaks", "unreq_leaks", "req_empty", "req_trunc", "observability", "json_strict", "json_semantic",
            "sort_strict", "sort_semantic", "arith_strict", "arith_semantic", "val_semantic_rate", "val_strict_rate")
    result = {"compositions": {m: {k: cells[m][k] for k in keep} | {"functionality": score.functionality(cells[m])}
                               for m in cells},
              "labels": {}}
    for tag, z in (("individual_95", score.Z95), ("postaudit_243", Z243)):
        r = score.classify(cells, 2, z=z)[3]
        result["labels"][tag] = {k: r[k] for k in ("security", "functionality", "contrasts",
                                                   "excess_point", "excess_lower", "excess_upper")}

    Path(args.out).write_text(json.dumps(result, indent=1))
    for m, c in result["compositions"].items():
        print(f"mask {m}: requested {c['req_leaks']}/100, unrequested {c['unreq_leaks']}, observability "
              f"{c['observability']:.2f}, validity {c['json_semantic'] + c['sort_semantic']}/60 semantic "
              f"({c['json_strict'] + c['sort_strict']}/60 strict), {c['functionality']}")
    for tag, r in result["labels"].items():
        print(f"A+B under {tag}: {r['security']} (contrasts {r['contrasts']}, smallest upper bound {r['excess_upper']:.3f})")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
