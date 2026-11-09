#!/usr/bin/env bash
# Recompute every number and figure in the paper from the archived raw outputs (CPU only).
#
#   bash reproduce.sh            verify: re-derive P1 and P2 analyses, re-run the confirmation
#                                simulation, redraw the figures, compare with the archive
#   bash reproduce.sh full       as above, and also re-run the synthetic screening sweeps
#
# Nothing in the archive is overwritten: everything is written to ./reproduced/ and compared.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-verify}"
OUT=reproduced
rm -rf "$OUT" && mkdir -p "$OUT"/{p1,real_audit,design,figures}
export PYTHONPATH="$PWD/p1/src:$PWD/analysis/src${PYTHONPATH:+:$PYTHONPATH}"
EV=p1/evidence/diagnostics
same() {  # same <reproduced> <archived> <label>
  if cmp -s "$1" "$2"; then echo "  identical: $3"; else echo "  DIFFERENT: $3 ($1 vs $2)"; exit 1; fi
}

echo "== tests"
python3 -m pytest -q p1/src/test_full64_cpu.py analysis/tests
(cd p2 && python3 tests/test_cpu.py | tail -1)

echo "== P1: re-verify the 19,200 Full64 and 800 G2-M records, conjunctive rule"
python3 analysis/src/audit_real.py --evidence p1/evidence --out "$OUT/real_audit" > "$OUT/real_audit/reverification.log"
same "$OUT/real_audit/full64_rescored.csv" analysis/results/real_audit/full64_rescored.csv "full64_rescored.csv"
same "$OUT/real_audit/audit_summary.json" analysis/results/real_audit/audit_summary.json "audit_summary.json"

echo "== P1: strict/semantic re-scoring and the asymmetric decision rule"
python3 analysis/src/rescore.py "$EV/full64_run_001" "$EV/full64_utility_bank.json" "$EV/g2m_bank.json" "$OUT/p1" > /dev/null
python3 analysis/src/decide.py "$OUT/p1" > "$OUT/p1/decide.log"
for f in rescored.csv requested_vectors.json decisions.json decision_sensitivity.csv monotonicity.csv; do
  same "$OUT/p1/$f" "analysis/results/p1/$f" "$f"
done

echo "== design bounds"
python3 analysis/src/cost_and_power.py --out "$OUT/design"
for f in analysis/results/design/*.csv; do same "$OUT/design/$(basename "$f")" "$f" "design/$(basename "$f")"; done

echo "== synthetic screening (13,140 runs)"
if [[ "$MODE" == full ]]; then
  python3 analysis/src/synthetic_screen.py --out "$OUT/simulated_main" --replicates 5
  python3 analysis/src/synthetic_screen.py --out "$OUT/simulated_noisy" --replicates 5 --noise
  python3 - <<'PY'
import sys; sys.path.insert(0, "analysis/src")
from synthetic_screen import simulate
for noise, name in [(False, "reproduced/synthetic_stress_clean"), (True, "reproduced/synthetic_stress_noisy")]:
    simulate(name, nvals=(20, 100), seeds=range(30),
             regimes=("monotone_pair", "order_mix", "antidote", "collapse_inhib", "benign"),
             extras=(64, 256), noise_fp=.01 if noise else 0, noise_fn=.03 if noise else 0)
PY
  for d in simulated_main simulated_noisy synthetic_stress_clean synthetic_stress_noisy; do
    same "$OUT/$d/summary.csv" "analysis/results/$d/summary.csv" "$d/summary.csv"
  done
else
  python3 - <<'PY'
import csv
for n, expect in [("simulated_main", 2160), ("simulated_noisy", 2160),
                  ("synthetic_stress_clean", 3600), ("synthetic_stress_noisy", 3600)]:
    with open(f"analysis/results/{n}/runs.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == expect, n
print("  archived sweeps: 11,520 runs present (run 'bash reproduce.sh full' to regenerate)")
PY
fi
python3 analysis/src/sim_confirm.py "$OUT/p1" > /dev/null
same "$OUT/p1/confirmation_summary.csv" analysis/results/p1/confirmation_summary.csv "confirmation_summary.csv (1,620 runs)"

echo "== figures (vector PDF)"
python3 analysis/src/make_figures.py analysis "$OUT" ieee "$OUT/figures/" analysis/results/p1 > /dev/null
ls "$OUT/figures"

echo "== P2: pre-registered replication"
python3 tools/verify_p2_archive.py

echo "REPRODUCTION ($MODE): PASS"
