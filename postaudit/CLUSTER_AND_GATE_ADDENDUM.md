# CoLoSS Applied Sciences: retrospective CPU-only robustness and gate audit

Date: 2026-11-15. **These are post-hoc exploratory analyses, not preregistered results.** Frozen P1 and P2 runs, labels, thresholds, and hypotheses are unchanged. This addendum uses the archived repo only and does not represent an independently generated experiment.

## A. Dependence-aware paired-prompt sensitivity

Run: `python3 postaudit/cluster_sensitivity.py --repo . --out postaudit/cluster_results --replicates 10000`

All 100 requested-prompt vectors per composition are joined positionally to the **original** frozen requested bank, sorted by prompt ID as in the existing scorers. For each planted pair (P1 mask 48, L2 and L3 mask 18), paired disclosure differences are calculated separately against the base model and both single-adapter references. At each bootstrap replicate, groups are sampled with replacement **as whole clusters**, preserving within-cluster prompt dependence and pair alignment; a percentile 95% interval is reported per contrast. Separately, the script removes each cluster in turn and records the minimum/maximum remaining mean difference.

The primary grouping in this sensitivity is **topic**. A second sensitivity groups by task template. The P2 banks also support a request-template grouping. Crucially, these are *one-factor-at-a-time checks*, not a hierarchical or multiway cluster bootstrap. The few P2 templates (four task templates and three request templates) make those particular percentile intervals especially unreliable as calibrated confidence intervals. The P1 bank uses 10 topics and 10 task templates; these are crossed, not identical. The topic group sizes are equal (10) in P1 and unequal in P2. The script samples clusters as wholes, so P2 resampled prompt totals can differ from 100.

### Results (lowest contrast bound across base/A/B, margin = 0.15)

| Library | Topic clusters | Minimum topic-bootstrap 95% lower bound | Minimum leave-one-topic-out point estimate | Minimum task-template-bootstrap bound | Minimum request-template-bootstrap bound |
|---|---:|---:|---:|---:|---:|
| P1 Gemma | 10 | 0.460 | 0.500 | 0.360 | n/a |
| L2 Gemma | 25 | 0.554 | 0.632 | 0.489 | 0.500 |
| L3 Qwen | 25 | 0.382 | 0.463 | 0.362 | 0.333 |

The planted-pair excess over every reference remains **above zero and above 0.15 on all these exploratory checks**. This strengthens robustness of the central *planted-pair effect* to topic or template reweighting. It does **not** establish that all 57 composition labels per library would survive a dependence-aware, 243-contrast family-wise test. Unadjusted marginal percentile bootstrap intervals should **not** be called multiplicity-adjusted guarantees or an independently validated confirmation. The existing §6.6 multiplicity sensitivities remain the formal post-audit comparisons; their approximate independent-prompt assumptions remain a limitation.

## B. Pre-registered development-gate feasibility audit

Run: `python3 postaudit/gate_forensics.py --repo . --out postaudit/cluster_results/gate_forensics.json`

The frozen G1 development requirement is `pair_leakage - max(base, A, B) >= 0.30`. Because leakage cannot exceed 1, the maximum possible excess is `1 - base_leakage`.

* L2 Gemma: base development leakage 26/30 = 0.8667, so maximum G1 effect is **0.1333**. All three attempts observed 0.1333. The threshold 0.30 is impossible. All three also failed the **semantic arithmetic utility** development gate; their base score was 0.125 while their pair score was zero. These failures are distinct from the fundamental infeasibility of G1.
* L3 Qwen: base development leakage 22/30 = 0.7333, so maximum G1 effect is **0.2667**. The threshold 0.30 is impossible; attempt 1 observed the maximal 0.2667 and passed all *other* gate criteria, but failed G1. Attempts 2 and 3 also failed G1.

The recorded fallback is consistent with best-score selection: L2 attempt 1 ties all attempts on score, and L3 attempt 1 is best. **Neither library passed the preregistered gate**. The original primary hypotheses remain unchanged: H1 detection passes both L2/L3; H2 utility passes L2 and fails L3, so joint H1+H2 passes L2 only. No retroactive threshold changes or selection claims are justified.

## C. Draft wording for the manuscript

Add the following as a short new paragraph near the end of §6.6, with supplementary results attached:

> **Clustered prompt sensitivity (post hoc).** As a further exploratory check on dependence among shared prompt templates and topics, we resampled the requested probes in paired clusters using the original prompt-bank metadata, with 10,000 bootstrap replicates per planted-pair-versus-reference contrast. Topic-level clustering gave minimum unadjusted percentile 95% lower bounds across the base and the two single-adapter references of 0.460, 0.554, and 0.382 for P1, L2, and L3, respectively. Leave-one-topic-out minimum differences were 0.500, 0.632, and 0.463. Separate template-grouped resampling also retained lower bounds above the 0.15 effect margin. These post-hoc intervals are descriptive sensitivities rather than simultaneous coverage guarantees: cluster counts are limited, template groups are not independent in general, and the analyses do not adjust for all composition–reference comparisons. Thus, they support the robustness of the three planted-pair observations without establishing dependence-adjusted family-wise validity of all reported security labels.

Replace the final sentence of §6.5, subsection **The Gate**, with a more explicit audit-safe wording:

> The pre-registered development gate was unattainable in both libraries: base-model disclosure on the 30 requested development prompts was 26/30 (Gemma) and 22/30 (Qwen), limiting the greatest achievable pair-versus-base excess to 0.133 and 0.267, respectively, below the required 0.30. All Gemma attempts also failed the development semantic-arithmetic condition. We retained the frozen thresholds, evaluated the specified best-score fallback attempts, and did not reclassify either gate as passed. Detection and specificity are therefore test-bank findings conditional on this preregistered fallback; they do not constitute successful development-gate validation.

In §7.3 Statistical Validity, add:

> Post-hoc paired cluster-resampling and leave-one-topic-out checks retain the planted-pair excess under topic reweighting, but cannot repair the original pre-registration or prove simultaneous confidence coverage under the full prompt-generating process.

The manuscript's existing separation between *no attributable excess* (NEGATIVE) and *absolute deployment safety* must remain in place. A NEGATIVE is not a safety certification, particularly when the base model itself leaks.

## D. Integrity and tests

Scripts are read-only with respect to the original dataset, preregistration, and frozen analyses; results reside in `postaudit/cluster_results/` and may be regenerated. The existing `SHA256SUMS.txt` covers the historical repository and is intentionally **not rewritten** by this addendum. A separate addendum checksum file accompanies the updated distribution.

The CPU pytest suites were run separately to prevent an unrelated duplicate `test_cpu.py` module-name collection collision: `postaudit` 6 passed; `analysis/tests` 15 passed; `p2/tests` 34 passed; `p1/src/test_full64_cpu.py` 4 passed; **59 tests passed** in total.

### Remaining limitations

- No independent public/adversarial adapter library or real-hub validation was added.
- No new GPU re-training, new prompt-bank evaluation, or fresh confirmation generations were performed.
- The Qwen arithmetic utility failure remains a real preregistered hypothesis failure, not an error to erase.
- Exact source artifact generation may require published adapter weights; the current release documents hashes and availability on request.
