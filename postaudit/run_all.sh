#!/usr/bin/env bash
# Post-audit CPU analyses (v1.0). Reads the frozen P1/P2 archives read-only; writes only to OUT.
#   bash postaudit/run_all.sh            -> writes postaudit/results_fresh/ and compares with postaudit/results/
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="${1:-$REPO/postaudit/results_fresh}"
PA="$REPO/postaudit"
mkdir -p "$OUT"
echo "== 1. corrected multiplicity inference (243/729 contrasts, strong margin, Hoeffding)"
python3 "$PA/final_inference.py" --repo "$REPO" --out "$OUT" > "$OUT/final_inference.log"
echo "== 2. corrected labels, scorer flags, retrospective replay (1,000 seeds per cell)"
python3 "$PA/run_postaudit.py" --repo "$REPO" --out "$OUT" --seeds 1000 > "$OUT/run_postaudit.log"
echo "== 3. semantic-only acceptances: structural categorisation of all 19,200 utility records"
python3 "$PA/semantic_deep_audit.py" --repo "$REPO" --out "$OUT" > "$OUT/semantic_deep_audit.log"
echo "== 4. structural sensitivity of functionality and security labels"
python3 "$PA/structural_label_impact.py" --repo "$REPO" --out "$OUT" > "$OUT/structural_label_impact.log"
echo "== 5. 10,000-replicate calibration of the NEGATIVE condition"
python3 "$PA/run_validation.py" --out "$OUT" > "$OUT/run_validation.log"
echo "== compare with archived postaudit/results"
status=0
for f in contrasts_243.csv decisions_reconciled.csv label_changes.csv hypotheses_reconciled.csv \
         decisions_corrected.csv replay_summary.csv all_semantic_only.csv utility_sensitivity.csv scorer_flags.csv; do
  if cmp -s "$OUT/$f" "$PA/results/$f"; then echo "  identical: $f"; else echo "  DIFFERENT: $f"; status=1; fi
done
python3 - "$OUT" "$PA/results" <<'PY'
import json,sys
from pathlib import Path
new,old=Path(sys.argv[1]),Path(sys.argv[2])
for f in ("inference_summary.json","decisions_summary.json","semantic_deep_summary.json","label_impact.json","calibration_results.json","scorer_summary.json"):
    a,b=json.loads((new/f).read_text()),json.loads((old/f).read_text())
    for k in ("method","caveat","notes","source_repository","protocol"):
        a.pop(k,None); b.pop(k,None)
    print(("  identical: " if a==b else "  DIFFERENT (content): ")+f)
PY
[ $status -eq 0 ] && echo "POST-AUDIT (verify): PASS" || { echo "POST-AUDIT (verify): FAIL"; exit 1; }
