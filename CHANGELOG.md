# Changelog

## v1.1 — 2026-11-15 (Applied Sciences submission; repository renamed CoLoSS)

- Repository renamed from `H1-LoRA-Screening` to `CoLoSS`; README, `CITATION.cff` and section
  references now follow the *Applied Sciences* manuscript.
- Added `postaudit/cluster_sensitivity.py` and `postaudit/gate_forensics.py` with archived outputs in
  `postaudit/cluster_results/`, tests, `CLUSTER_AND_GATE_ADDENDUM.md` and `ADDENDUM_SHA256SUMS.txt`:
  post hoc paired cluster bootstrap (10,000 replicates) and leave-one-cluster-out checks of the planted
  pairs, and a feasibility audit of the P2 development gate. Exploratory; not pre-registered.
- Added `external_validation/` (P3): GPU runner `run_external.py`, `report.py`, the four archived P3
  runs with provenance (adapter Hub commits and SHA-256 digests), `verify_p3.py` (CPU record check) and
  `apply_frozen_rule.py`, which applies the frozen P2 scorer and rule unmodified and writes
  `p3_frozen_rule.json`.
- `postaudit/README.md`: new scripts and current table numbers.
- Added `RELEASE_SHA256SUMS.txt` covering every file of this release. `SHA256SUMS.txt` (v1.0) and
  `postaudit/ADDENDUM_SHA256SUMS.txt` are kept unchanged; the addendum verifies in full, and every code
  and data entry of `SHA256SUMS.txt` verifies (only the six documentation and configuration files listed in
  the README were edited since v1.0).
- Adapter weights of P1 and P2 (both libraries, all three training attempts) archived on Zenodo
  (https://doi.org/10.5281/zenodo.24261411), with `tools/package_weights.sh`, which verifies every file against `p1/` and `p2/adapter_hashes.txt`
  before packaging. Previously available on request.
- Added `RELEASE_AUDIT.md`: an independent release-readiness audit of the v1.0-based package and its
  re-check on v1.1. `external_validation/README.md` now sets up a virtual environment before installing.
- Frozen and unchanged: `p1/`, `p2/`, all raw generations, archived labels and analysis outputs.
  `reproduce.sh full` → `REPRODUCTION (full): PASS`; `postaudit/run_all.sh` → `POST-AUDIT (verify): PASS`.

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

- Initial release (prepared for an earlier submission to IEEE TDSC): P1 exhaustive study, P2 pre-registered
  replication, synthetic screening study, CPU reproduction.
