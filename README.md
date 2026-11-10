# Composition-aware LoRA Safety Screening

Code, prompt banks, raw generations and analysis for the paper

> U. Yavanoğlu, *When Silence Looks Safe: Output Collapse and Evaluation Bias in Composition-Aware Safety Screening of LoRA Adapters*, submitted to REDACTED

Every number and figure in the paper can be recomputed on a CPU from the files in this repository in under a minute:

```bash
pip install -r requirements-cpu.txt
bash reproduce.sh          # re-derives P1 and P2, redraws figures, compares with the archive
bash reproduce.sh full     # also re-runs the synthetic screening sweeps (about 40 s)
bash postaudit/run_all.sh  # post-audit sensitivity analyses of Section 6.6 (about 6 s)
```

The script writes to `reproduced/`, compares every regenerated file with the archived one, and ends with `REPRODUCTION (verify): PASS`.

## What is here

| Folder | Part of the paper | Contents |
|---|---|---|
| `p1/` | P1: exhaustive diagnostic study (Sections 5.1–5.4, RQ1–RQ3) | Training, operator and generation code (GPU); library config; the G2-M boundary bank and utility bank; all 19,200 Full64 generations and the 800 G2-M generations with manifests; adapter hashes |
| `analysis/` | P1 re-analysis, synthetic study (RQ4), figures | CPU scripts; archived results (`results/`); figures (`figures/`); tests |
| `p2/` | P2: pre-registered replication (Section 5.5, RQ5) | The frozen replication package exactly as it ran, the pre-registration, prompt banks, training records, development-gate records, all 38,400 test generations, analysis outputs and logs |
| `tools/` | | `verify_p2_archive.py`, an independent check of the P2 archive |
| `postaudit/` | Post-audit sensitivity analyses (Section 6.6, Table 3, Supplement S5) | CPU scripts and archived outputs: corrected 243/729-contrast intervals, strong-margin and Hoeffding sensitivities, structural audit of the semantic scorer, retrospective replay of budgeted policies, calibration of the NEGATIVE condition; `run_all.sh` reproduces all of it in a few seconds |

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
- `logs/`: console logs of training, gating, Full64 and analysis.
- `RUN_GUIDE.md` (how it was run), `REVIEW_RESPONSE.md` (independent code reviews and fixes), `adapter_hashes.txt`.

The Full64 manifests record the absolute path of the library on the workstation, so `p2.analyze` itself only runs in place; `tools/verify_p2_archive.py` performs the same checks without the machine-specific path and re-runs the frozen analysis.

## Re-running the GPU experiments

The GPU parts need an NVIDIA GPU with at least 24 GB (we used an RTX 5090, 32 GB), PyTorch with CUDA, and Hugging Face access to the gated Gemma model (`pip install -r requirements-gpu.txt`, `hf auth login`).

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

Greedy generation was deterministic on our hardware (the G2-M and Full64 runs agree output for output), but re-runs on other GPUs or library versions may differ in individual tokens; the archived outputs are the evidence the paper reports.

## Post-audit analyses (`postaudit/`, manuscript v1.0)

After the v0.9 draft, an independent audit of the analysis code found that its family-wise check used
`z = 3.195`, the one-sided 0.05/57 quantile. The manuscript now reports two regimes and keeps them
apart: the **frozen regime** (individual 95% intervals; the pre-registered analysis, unchanged) and the
**post-audit regime**, which controls all 243 composition–reference contrasts per library
(`z = 3.711836`), adds a 729-contrast cross-library family, a strong-margin criterion and a
finite-sample Hoeffding bound, audits the semantic utility scorer, and replays four budgeted
screening policies on the real libraries. `postaudit/README.md` documents the scripts, outputs and
caveats. The post-audit family was not pre-registered; it is a sensitivity analysis.

Two consequences for the frozen parts of this repository:

- `analysis/src/decide.py` (the P1 CPU analysis, not part of the digest-protected P2 package) now
  computes its sensitivity grid with the two-sided values `z = 3.327213` (57 compositions) and
  `z = 3.711836` (243 contrasts) and adds a strong-margin column. Its main output, the frozen
  labels in `analysis/results/p1/decisions.json`, is byte-identical to v0.9; only
  `decision_sensitivity.csv` changed, and the v0.9 grid is kept as
  `decision_sensitivity_v09_z3.195.csv`. `reproduce.sh` still ends with `PASS`.
- `p2/p2/score.py` still contains `ZBONF = 3.195`. It is frozen by the pre-registration digest and
  is not edited; the paper reports its Bonferroni hypothesis variants only as frozen outcomes.

## Adapter weights

The adapter weights, including the colluding pairs, are not distributed in this repository. They are available from the author on reasonable request (uraz@gazi.edu.tr). Their SHA-256 hashes are in `p1/adapter_hashes.txt` and `p2/adapter_hashes.txt`, so received weights can be checked. The colluding adapters disclose only random canary strings and were trained on no harmful content.

## Environment

Ubuntu 24.04 under WSL2, Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 5090 (32 GB) provided by UAYTECH. The CPU reproduction needs only the packages in `requirements-cpu.txt`.

## Changes

See `CHANGELOG.md`.

## Licence and citation

Code is released under the MIT licence and data (prompt banks, generations, analysis outputs) under CC BY 4.0; see `LICENSE`. Please cite the paper; see `CITATION.cff`.

## Funding

This work was financially supported by UAYTECH Corporation, which also provided the GPU workstation.

## AI Assistance Disclosure

OpenAI GPT and Anthropic Claude models assisted with code review, debugging, and validation. All research findings and conclusions remain the responsibility of the author.
