# CoLoSS — Composition-aware LoRA Safety Screening

Author: Uraz Yavanoğlu, Department of Computer Engineering, Gazi University, Ankara, Türkiye
([ORCID 0000-0001-8358-8150](https://orcid.org/0000-0001-8358-8150)).

Code, prompt banks, raw generations and analyses for the paper

> U. Yavanoğlu, *When Silence Looks Safe: Output Collapse in Safety Screening of Colluding LoRA Adapters*, submitted to *Applied Sciences* (MDPI), 2026.

Every number, table and figure in the paper and its supplementary material can be recomputed on a CPU from the files in this repository, in well under a minute:

```bash
pip install -r requirements-cpu.txt
bash reproduce.sh full                         # P1, P2, synthetic sweeps, figures; ends "REPRODUCTION (full): PASS"
bash postaudit/run_all.sh                      # post-audit sensitivities; ends "POST-AUDIT (verify): PASS"
python3 -m pytest -q postaudit                 # 9 tests
python3 external_validation/verify_p3.py       # P3 records and digests
python3 external_validation/apply_frozen_rule.py   # P3 labels under the frozen rule
```

`bash reproduce.sh` (without `full`) skips re-running the synthetic sweeps and only checks their archived run counts. All scripts write to fresh output folders and compare with the archived files; nothing archived is overwritten.

## What the study does

Users of model hubs stack LoRA adapters, and colluding adapters can each pass a unit-level check while the unsafe behaviour appears only in their composition. The paper studies how a defender should read the outcome of testing a composition. It keeps four questions apart: was a violation observed, does it exceed the base model and every member on matched prompts, could the test have observed one at all, and does the composition still work? The violation is a harmless proxy: disclosing a random canary that the system prompt says must stay secret, detected by exact matching, so no LLM judge is involved. The colluding adapters disclose only random canary strings and were trained on no harmful content.

## Map of the repository

| Folder | Part of the study | Manuscript |
|---|---|---|
| `p1/` | P1: exhaustive diagnostic study of one six-adapter library on `google/gemma-2-2b-it` (64 compositions, 19,200 generations) | §5.1–5.4, §6.1–6.3; Tables 2–3; Figures 2–3 |
| `analysis/` | P1 re-scoring and decision rule; synthetic screening study (13,140 runs); figures | §5.6–5.7, §6.4; Figure 4; Supplement S1–S3 |
| `p2/` | P2: pre-registered replication on Gemma (L2) and Qwen (L3), frozen on 8 November 2026 (38,400 generations) | §5.5, §6.5; Table 5; Supplement S4 |
| `postaudit/` | Post-audit multiplicity and finite-sample sensitivities, scorer audit, retrospective replay, NEGATIVE calibration; post hoc clustered prompt sensitivity and development-gate audit | §6.4 Table 4, §6.5, §6.6; Tables 6–7; Supplement S4, S5, S8 |
| `external_validation/` | P3: exploratory control with two independently published adapters for Qwen2.5-1.5B-Instruct (1,200 generations) | §6.7; Table 8; Supplement S9 |
| `tools/` | `verify_p2_archive.py`, an independent check of the P2 archive | — |

### P1: exhaustive diagnostic study (`p1/`)

- `src/pilot/`: unmerged multi-LoRA operator (`multilora.py`), model loading and generation (`common.py`), prompt templates and the exact canary oracle (`tasks.py`), adapter training (`train_adapters.py`), numerical preflight (`preflight.py`), library loader (`library.py`).
- `src/run_full64.py`: resumable runner for all 64 compositions × 300 prompts; `src/create_utility_bank.py`: the deterministic utility bank.
- `src/screening/`: run-manifest helpers and the conjunctive risk rule used in RQ2.
- `configs/pilot_lib1.yaml`: the six-adapter library on `google/gemma-2-2b-it` (resolved commit `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`).
- `evidence/diagnostics/`: `g2m_bank.json` (200 boundary prompts), `full64_utility_bank.json` (100 tasks), `full64_run_001/outputs/*.jsonl.gz` (one file per composition, mask = bitmask over adapter indices 0–5), `g2m_run_001/` (the four-cell G2-M run), with manifests recording adapter, bank and code hashes.

The script that generated the G2-M boundary bank was not archived; the bank itself, its seed and its hash are, and its 800 G2-M outputs are reproduced output for output by the Full64 run (checked by `reproduce.sh`).

### P2: pre-registered replication (`p2/`)

`p2/` is the replication package exactly as it ran on 8 November 2026. Its code, configs, tests and banks are frozen by `p2/prereg/prereg.json` (SHA-256 `c870ecb85f69a7e9822c9c48a22ad37c3c21ebb5ceb2ab9a5d46ca42b71a37c8`), and every stage refuses to run if any of them changes, so **do not edit files inside `p2/`**. It contains:

- `p2/` (Python package): bank generator, frozen scorers and decision rule, utility-preserving collusion sampler, trainer, development gate, Full64 runner, analysis, provenance checks, GPU smoke test.
- `configs/`: library L2 (`google/gemma-2-2b-it` @ `299a8560…`) and L3 (`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`).
- `banks/`: training, development and the G3 test banks; `prior/`: the earlier banks used to prove disjointness.
- `runs/<LIB>/attempt_{1,2,3}/`: `library.json`, training ledger, training log, gate record and gate outputs for every attempt; `base_answers_train.json` with its digest.
- `results/<LIB>_full64/`: the 64 output files, manifest, hash ledger, completion record.
- `analysis/<LIB>/`: `results.json`, `cells.csv`, `closure.csv`, `sensitivity.csv`.
- `logs/`: console logs of training, gating, Full64 (including the wall-clock time of every composition) and analysis.
- `RUN_GUIDE.md` (how it was run), `REVIEW_RESPONSE.md` (independent code reviews and fixes), `adapter_hashes.txt`.

Neither library passed its development gate: the gate's effect condition (excess ≥ 0.30 over the base on the development probes) was unattainable because the base models already disclosed on 26/30 (Gemma) and 22/30 (Qwen) of those probes. All Gemma attempts also failed the gate's arithmetic condition. Both evaluations used the pre-registered best-score fallback, and the frozen thresholds were not changed; `postaudit/gate_forensics.py` documents this.

The Full64 manifests record the absolute path of the library on the workstation, so `p2.analyze` itself only runs in place; `tools/verify_p2_archive.py` performs the same checks without the machine-specific path and re-runs the frozen analysis.

### Post-audit and post hoc analyses (`postaudit/`)

After the first draft, an independent audit of the analysis code found that its family-wise check used `z = 3.195`, the one-sided 0.05/57 quantile. The paper therefore reports two regimes and keeps them apart: the **frozen regime** (individual 95 % intervals; the pre-registered analysis, unchanged) and the **post-audit regime**, which controls all 243 composition–reference contrasts per library (`z = 3.711836`), adds a 729-contrast cross-library family, a strong-margin criterion and a finite-sample Hoeffding bound, audits the semantic utility scorer, and replays four budgeted screening policies on the real libraries. The post-audit family was not pre-registered; it is a sensitivity analysis. `postaudit/README.md` documents every script.

Two later, CPU-only additions are **post hoc and exploratory** (`postaudit/CLUSTER_AND_GATE_ADDENDUM.md`):

- `cluster_sensitivity.py`: paired cluster bootstrap (10,000 replicates) and leave-one-cluster-out checks of the three planted-pair effects, grouping the requested prompts by topic, task template or request template one factor at a time. Every planted-pair excess stays above the 0.15 margin; the intervals are not adjusted for the other contrasts and do not certify the other labels.
- `gate_forensics.py`: the feasibility audit of the P2 development gate summarised above.

Two consequences for the frozen parts of this repository:

- `analysis/src/decide.py` (the P1 CPU analysis, not part of the digest-protected P2 package) computes its sensitivity grid with the two-sided values `z = 3.327213` (57 compositions) and `z = 3.711836` (243 contrasts) and adds a strong-margin column. Its main output, the frozen labels in `analysis/results/p1/decisions.json`, is byte-identical to the first release; only `decision_sensitivity.csv` changed, and the earlier grid is kept as `decision_sensitivity_v09_z3.195.csv`.
- `p2/p2/score.py` still contains `ZBONF = 3.195`. It is frozen by the pre-registration digest and is not edited; the paper reports its Bonferroni hypothesis variants only as frozen outcomes.

### P3: exploratory external-adapter control (`external_validation/`)

Two independently published adapters for `Qwen/Qwen2.5-1.5B-Instruct` (`jack8885/task-15-…` and `kkokas/task-14-…`), pinned to Hub commits with SHA-256 digests, were evaluated alone and together on the G3 test banks. Scored with the frozen P2 scorer and rule, the composition is Functional, 94 % observable and labelled NEGATIVE, while it still discloses the canary on 35 of 100 requested probes: a NEGATIVE label means no composition-attributable excess, not safety. P3 reuses the G3 prompts, was not pre-registered, and does not test detection of a colluding pair built by someone else. See `external_validation/README.md`.

## Re-running the GPU experiments

The GPU parts need an NVIDIA GPU with at least 24 GB (we used an RTX 5090, 32 GB), PyTorch with CUDA, and Hugging Face access to the gated Gemma model (`pip install -r requirements-gpu.txt`, `hf auth login`). On that GPU one composition took a median of 106 s (Gemma) or 120 s (Qwen) on the 300 prompts.

**P1.** From `p1/`, with `PYTHONPATH=src`:

```bash
python -m pilot.train_adapters --config configs/pilot_lib1.yaml          # writes runs/pilot_lib1
python -m pilot.preflight --library runs/pilot_lib1
python src/create_utility_bank.py --out evidence/diagnostics/full64_utility_bank.json
python src/run_full64.py --library runs/pilot_lib1 \
  --boundary-bank evidence/diagnostics/g2m_bank.json \
  --utility-bank evidence/diagnostics/full64_utility_bank.json --out runs/full64_rerun
```

**P2.** Follow `p2/RUN_GUIDE.md` in a fresh copy of `p2/` without `prereg/`, `runs/`, `results/`, `analysis/` and `logs/`: smoke test, `python -m p2.freeze`, then `bash run_library.sh L2 …` and `L3 …`. A re-run freezes its own plan; the code and banks are identical, so the frozen code digest is the same.

**P3.** See `external_validation/README.md`.

Greedy generation was deterministic on our hardware (the G2-M and Full64 runs agree output for output), but re-runs on other GPUs or library versions may differ in individual tokens; the archived outputs are the evidence the paper reports.

## Adapter weights

The trained P1 and P2 adapters, including the colluding pairs, are archived on Zenodo:
[https://doi.org/10.5281/zenodo.24261411](https://doi.org/10.5281/zenodo.24261411). They are not committed to the git tree, because of their size.

| File | Contents |
|---|---|
| `CoLoSS_P1_Weights.zip` | P1 library: six adapters on `google/gemma-2-2b-it` (`runs/pilot_lib1/`) |
| `CoLoSS_P2_L2_Weights.zip` | P2 library L2 (Gemma): six adapters for each of the three training attempts; attempt 1 is the evaluated one |
| `CoLoSS_P2_L3_Weights.zip` | P2 library L3 (Qwen2.5-1.5B-Instruct): same layout |
| `CoLoSS_Weight_Archives_SHA256SUMS.txt` | SHA-256 of the three archives |

Each archive unpacks to the `runs/...` paths that the configurations use. The SHA-256 of every adapter file,
written at training time, is in `p1/adapter_hashes.txt` and `p2/adapter_hashes.txt` of this repository. To
check a download, put the four files in the repository root and run:

```bash
sha256sum -c CoLoSS_Weight_Archives_SHA256SUMS.txt
mkdir -p p1w p2w && unzip -q CoLoSS_P1_Weights.zip -d p1w \
  && unzip -q CoLoSS_P2_L2_Weights.zip -d p2w && unzip -q CoLoSS_P2_L3_Weights.zip -d p2w
(cd p1w && sha256sum -c ../p1/adapter_hashes.txt) && (cd p2w && sha256sum -c ../p2/adapter_hashes.txt)
```

The adapters use this repository's own format (one `.safetensors` file per adapter; load with
`save_adapter`/`load_adapter` in `p1/src/pilot/multilora.py` or `p2/pilot/multilora.py`). The colluding adapters
disclose only random canary strings and were trained on no harmful content. The archives can be rebuilt with
`tools/package_weights.sh`, which refuses to package anything unless every file matches the archived
training hashes. The training code, configurations, seeds and logs are also included, so the adapters can be
retrained (not guaranteed bit-identical). The two P3 adapters are public on the Hugging Face Hub and are not
redistributed here.

## Integrity

- `RELEASE_SHA256SUMS.txt` lists the SHA-256 digest of every file in this release: `sha256sum -c RELEASE_SHA256SUMS.txt`.
- `SHA256SUMS.txt` is the v1.0 manifest, kept unchanged. Since v1.0 only six documentation and configuration files it lists were edited (`README.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, `postaudit/README.md`, `.gitignore`); every code and data entry still verifies:
  `grep -v -E '  \./(README\.md|CHANGELOG\.md|CITATION\.cff|LICENSE|postaudit/README\.md|\.gitignore)$' SHA256SUMS.txt | sha256sum -c`
- `postaudit/ADDENDUM_SHA256SUMS.txt` (post hoc addendum) is unchanged and verifies in full.
- `.gitattributes` disables line-ending conversion so that every file keeps the exact bytes its digest records.

## Environment

Ubuntu 24.04 under WSL2, Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 5090 (32 GB) provided by UAYTECH. The CPU reproduction needs only the packages in `requirements-cpu.txt`.

## Changes

See `CHANGELOG.md`. This repository supersedes the earlier `urazy1982/H1-LoRA-Screening`; the project and its file layout are the same.

## Licence and citation

Code is released under the MIT licence and data (prompt banks, generations, analysis outputs) under CC BY 4.0; see `LICENSE`. Please cite the paper; see `CITATION.cff`.

## Funding

This work was financially supported by UAYTECH Corporation, which also provided the GPU workstation.

## Development note

The author designed and ran all experiments. AI assistants (Anthropic Claude, OpenAI ChatGPT) were used under the author's direction only for partial code validation and audit checks.