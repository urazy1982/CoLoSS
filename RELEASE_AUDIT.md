# Independent release-readiness audit — 2026-11-15

## Tests actually performed

- Original submitted ZIP: archive CRC PASS.
- Original root SHA256SUMS: PASS for 408 listed entries; NOTE that the original checksum list omitted new P3/cluster files. This release regenerates the complete list.
- `python3 -m pytest -q p1/src/test_full64_cpu.py analysis/tests postaudit/test_cluster_sensitivity.py postaudit/test_final_inference.py`: 28 passed.
- `bash reproduce.sh` (standard verification mode): PASS, including P1 outputs, P2 frozen records, confirmation simulations, and figure regeneration.
- `bash postaudit/run_all.sh`: PASS, byte-identical/comparable archived statistical inferences and replay summaries.
- `external_validation/apply_frozen_rule.py`: PASS; independently regenerated JSON was byte-identical to archived `p3_frozen_rule.json`.
- `postaudit/gate_forensics.py` and `postaudit/cluster_sensitivity.py --replicates 2000`: executed successfully.
- `bash reproduce.sh full`: attempted, but full synthetic sweep exceeded available execution time after the first two synthetic sweeps. Do NOT claim an independent PASS for full sweep. Its archived results were checked in the standard mode.

## P3 re-analysis

Base: 38/100 requested leaks, 48/60 semantic validity. A: 28/100, 47/60. B: 49/100, 51/60. A+B: 35/100, 46/60; functionality Functional, observability 0.94. A+B NEGATIVE under individual 95% and post-audit 243-contrast rule; post-audit minimum upper CI approximately 0.0435. NEGATIVE is *not* a deployment safety clearance.

## Remaining limitations

- The original trained P1/P2 adapter weights are not supplied, so generation/training cannot be independently replayed with exactly those weights. CPU rescoring of archived generations is reproducible.
- P3 reuses G3 prompts and is an exploratory external-source control, not external collusion detection validation.
- Current checks do not independently establish that all Hugging Face model/adapter weights remain publicly downloadable or license-compatible.
- No fresh GPU execution was performed during this audit.
- No GitHub push, Zenodo deposit, or DOI registration was performed.

## Release repairs

- Updated outdated IEEE journal reference in README and CITATION.cff to reflect manuscript preparation for Applied Sciences (without claiming journal acceptance).
- Updated P3 environment instructions to use Python virtualenv, avoiding PEP 668.
- Removed disposable caches and locally regenerated outputs.
- Regenerated SHA256SUMS across all released files, including previously omitted P3 and cluster assets.
- Left frozen P1/P2 experimental data, protocols and analysis source unchanged.

---

## v1.1 re-check — 2026-11-15

The audit above was run on a v1.0-based package. Its code and data files are byte-identical to v1.1; only
documentation and metadata differ (repository name, manuscript title, section and table numbers, the
adapter-weight release, and the P3 CPU checker `external_validation/verify_p3.py`). The checks were repeated on
the v1.1 tree:

- `sha256sum -c RELEASE_SHA256SUMS.txt` (every file of the release): PASS.
- `sha256sum -c postaudit/ADDENDUM_SHA256SUMS.txt` from the repository root: PASS.
- `python3 -m pytest -q p1/src/test_full64_cpu.py analysis/tests postaudit`: 28 passed.
- `cd p2 && python3 -m pytest -q` (the frozen P2 package's own suite): 34 passed; 62 tests in total.
- `bash reproduce.sh`: `REPRODUCTION (verify): PASS` (about 13 s).
- `bash reproduce.sh full`, including the synthetic sweeps: `REPRODUCTION (full): PASS` (about 25 s on the
  machine used for this re-check).
- `bash postaudit/run_all.sh`: `POST-AUDIT (verify): PASS`.
- `python3 external_validation/verify_p3.py`: PASS for all four P3 runs; `apply_frozen_rule.py` reproduces
  `p3_frozen_rule.json` byte for byte.
- Running all of the above left every file in `RELEASE_SHA256SUMS.txt` unchanged.

Differences from the repairs listed above:

- Checksums: v1.1 keeps `SHA256SUMS.txt` exactly as released with v1.0, so that the v1.0 record stays verifiable,
  and adds `RELEASE_SHA256SUMS.txt`, which covers every file of v1.1.
- Adapter weights: the P1 and P2 adapters are archived on Zenodo (https://doi.org/10.5281/zenodo.24261411; see README,
  "Adapter weights"), built with `tools/package_weights.sh`, which refuses to package any file that does not
  match `p1/adapter_hashes.txt` and `p2/adapter_hashes.txt`. With them, the evaluation runs can be replayed
  with the original weights on a GPU.
- The other limitations above still apply: no fresh GPU execution was part of either check, and P3 remains
  an exploratory reuse of the G3 prompts.
