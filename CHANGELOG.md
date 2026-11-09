# Changelog

## v1.0 — 2026-11-09 (post-audit reconciliation)

- Added `postaudit/`: `final_inference.py` (243/729-contrast Bonferroni–Newcombe, strong-margin and
  Hoeffding sensitivities over the 192 archived compositions), `run_postaudit.py` (corrected labels,
  scorer flags, retrospective replay of four budgeted policies, 72 cells × 1,000 seeds),
  `semantic_deep_audit.py` and `structural_label_impact.py` (393 semantic-only acceptances, 54 unclosed
  JSON objects, label impact), `run_validation.py` (10,000-replicate calibration of the NEGATIVE
  condition), `run_all.sh`, tests, archived outputs in `postaudit/results/`, and
  `docs/INFERENCE_ADDENDUM.pdf`.
- `analysis/src/decide.py`: sensitivity grid uses two-sided Bonferroni values 3.327213 (57) and
  3.711836 (243) instead of 3.195 (one-sided 0.05/57) and adds a strong-margin column.
  `analysis/results/p1/decision_sensitivity.csv` regenerated; previous grid kept as
  `decision_sensitivity_v09_z3.195.csv`. `decisions.json` and `monotonicity.csv` unchanged.
- `README.md`: post-audit section; `CITATION.cff`: release date; `SHA256SUMS.txt` added.
- Frozen: `p1/`, `p2/`, raw generations, archived labels. `reproduce.sh` → `REPRODUCTION (verify): PASS`.

## v0.9 — 2026-11-08

- Initial release accompanying the TDSC submission: P1 exhaustive study, P2 pre-registered
  replication, synthetic screening study, CPU reproduction.
