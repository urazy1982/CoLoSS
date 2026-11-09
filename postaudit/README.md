# Post-audit analyses (manuscript v1.0, Section 6.6 and Supplement S5)

CPU-only re-analysis of the archived P1 and P2 outputs, added after an independent audit of the
analysis code found that the family-wise check in the v0.9 draft used an incorrect critical value
(`z = 3.195`, the one-sided 0.05/57 quantile). Nothing in `p1/`, `p2/` or the archived
`analysis/results/` raw records is modified by anything in this directory; the scripts read the
frozen archives and write only to the output directory you name.

These analyses are **sensitivity analyses**, not pre-registered tests. The frozen regime of the paper
(individual 95 % paired Newcombe intervals, `p2/p2/score.py`, `analysis/src/decide.py` default run) is
the historical record and is reported unchanged; the post-audit regime is reported beside it.

## Run

```bash
pip install -r requirements-cpu.txt          # numpy, scipy only needed by run_validation.py
bash postaudit/run_all.sh                    # ~6 s on CPU; writes postaudit/results_fresh/ and
                                             # compares every file with postaudit/results/
python3 -m pytest -q postaudit               # 6 tests
```

Expected last line: `POST-AUDIT (verify): PASS`.

## What each script does

| Script | Reads | Writes | Paper |
|---|---|---|---|
| `final_inference.py` | requested-probe vectors of all 192 compositions (`analysis/results/p1/requested_vectors.json`, `p2/analysis/*/requested_vectors.json`) and the archived labels | `contrasts_243.csv` (729 rows: every composition–reference contrast with 243-adjusted bounds), `decisions_reconciled.csv` (192 rows: frozen label and every post-audit label), `label_changes.csv`, `hypotheses_reconciled.csv`, `inference_summary.json` | §6.6, Tables 5–6, Supplement S5 |
| `run_postaudit.py` | same vectors; the 19,200 utility records | `decisions_corrected.csv`, `decisions_summary.json` (57/243 families), `scorer_flags.csv`, `scorer_summary.json`, `replay_summary.csv` (4 policies × 3 budgets × 2 label sets × 3 libraries, 1,000 seeds per cell), `status.json` | §6.4 Table 3, Supplement Table S4 |
| `semantic_deep_audit.py` | the 19,200 utility records, scored with the frozen P2 scorer (`p2/p2/score.py`) as a common diagnostic | `all_semantic_only.csv` (393 responses accepted semantically and rejected strictly, with category), `semantic_deep_summary.json` | §6.6 "Structural audit" |
| `structural_label_impact.py` | the above | `utility_sensitivity.csv`, `label_impact.json` (functionality and security labels when credit is withheld from the 54 unclosed JSON objects) | §6.6 |
| `run_validation.py` | nothing from the project | `calibration_results.json` (10,000 synthetic replicates of the NEGATIVE condition at the true-excess boundary) | §6.6 "What the frozen NEGATIVE condition risks" |

## Critical values and procedures

Two-sided Bonferroni quantiles at family α = 0.05: individual 1.959964; 57 compositions 3.327213;
243 contrasts per library 3.711836; 729 contrasts across the three libraries 3.981135. Hoeffding
half-width for the mean of 100 paired differences in [−1, 1] at 243 simultaneous contrasts:
0.428531. The family of 243 is Σ_{|S|≥2}(1+|S|) over the 57 multi-adapter compositions of a
six-adapter library: each composition is compared with the base model and each of its members.

Procedures (all with δ = 0.15, τ_o = 0.90, the frozen POSITIVE/NEGATIVE/INCONCLUSIVE rule):

- `original`: individual 95 % intervals (the frozen regime; reproduces all 192 archived labels, 0 mismatches)
- `family57`, `family243`, `family729`: the same rule with Bonferroni-widened intervals
- `strong_margin243`: POSITIVE additionally requires every adjusted lower bound > δ
- `hoeffding_243`, `hoeffding_strong_margin`: Newcombe interval replaced by Δ̂ ± 0.4285 (valid only if the 100 prompt outcomes are independent)

## Headline results (from `results/inference_summary.json`)

| Library | frozen P/N/I | 243 | 729 | strong margin | Hoeffding | planted pair |
|---|---|---|---|---|---|---|
| P1 | 12/3/42 | 10/3/44 | 9/3/45 | 8/3/46 | 5/3/49 | POSITIVE in all but Hoeffding full-margin |
| L2 | 16/24/17 | 16/23/18 | 16/23/18 | 16/23/18 | 8/21/28 | POSITIVE in all |
| L3 | 16/27/14 | 14/27/16 | 14/27/16 | 9/27/21 | 8/18/31 | POSITIVE in all but Hoeffding full-margin |

Closure (POSITIVE strict supersets of the pair, of 15): L2 15 → 15 → 15; L3 15 → 13 → 8 (frozen → 243 → strong margin).
No procedure produces a POSITIVE composition that does not contain the planted pair.

## Caveats, as stated in the paper

- The 243-contrast family was declared for the post-audit analysis; it was not pre-registered.
- Bonferroni widening makes the paired Newcombe intervals conservative for the family but does not make their coverage exact.
- The Hoeffding bound assumes independent prompt outcomes; the banks are built from shared templates and topics.
- The replay uses cached labels on three six-adapter libraries and involves no fresh generations or confirmation.
- The structural audit is a format criterion applied programmatically; it is not a human adjudication of semantic correctness.
- `p2/p2/score.py` still contains `ZBONF = 3.195`; it is part of the frozen, digest-protected replication package and is not edited. Its "Bonferroni" hypothesis variants are reported in the paper only as frozen outcomes.

`docs/INFERENCE_ADDENDUM.pdf` is the two-page typeset summary of these results.
