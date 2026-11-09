#!/usr/bin/env bash
# Train -> gate -> (next attempt if the gate fails) -> Full64 -> analysis, for one library.
# Usage: bash run_library.sh L2 configs/L2_gemma2_2b.yaml
set -euo pipefail
mkdir -p logs
LIB="$1"; CONFIG="$2"; BS="${BS:-16}"
python tests/test_cpu.py
N=$(python - "$CONFIG" <<'PY'
import sys, yaml; print(len(yaml.safe_load(open(sys.argv[1]))["collusion_attempts"]))
PY
)
gate_pass() { python - "$1" <<'PY'
import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)
PY
}
for ATT in $(seq 1 "$N"); do
  D="runs/$LIB/attempt_$ATT"
  if [ ! -f "$D/library.json" ]; then
    python -m p2.train_library --config "$CONFIG" --attempt "$ATT" 2>&1 | tee -a "logs/${LIB}_train_a$ATT.log"
  fi
  if [ ! -f "$D/gate.json" ]; then
    python -m p2.gate --library "$D" --batch-size "$BS" 2>&1 | tee -a "logs/${LIB}_gate_a$ATT.log"
  fi
  if gate_pass "$D/gate.json"; then echo "attempt $ATT passed the gate"; break; fi
  echo "attempt $ATT failed the gate"
done
SEL=$(python -m p2.run_full64 --select "runs/$LIB")
echo "selected: $SEL"
python -m p2.run_full64 --library "$SEL" --out "results/${LIB}_full64" --batch-size "$BS" 2>&1 | tee -a "logs/${LIB}_full64.log"
python -m p2.analyze --run "results/${LIB}_full64" --library "$SEL" --out "analysis/$LIB" | tee "logs/${LIB}_analysis.txt"
