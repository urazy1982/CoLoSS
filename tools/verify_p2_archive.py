#!/usr/bin/env python3
"""Independent, CPU-only verification of the archived P2 replication (no GPU, no weights).

Checks, from the files in p2/:
  1. prereg/prereg.json has the published SHA-256, and the code, configs, prior record and
     all seven prompt banks match the digests frozen in it (p2.integrity.verify_frozen);
  2. for each library, every Full64 output file matches the hash ledger written at
     generation time, complete.json matches manifest and ledger, and every record re-validates
     (prompt order, prompt hashes, oracle labels);
  3. the evaluated attempt is the one the pre-registered selection rule picks, and the
     manifest's library.json, ledger and gate digests match the archived files (the absolute
     library path in the manifest is machine-specific and is not compared);
  4. adapter_hashes.txt (weights are not distributed) matches each training ledger;
  5. the frozen analysis re-run from the raw outputs reproduces analysis/<LIB>/results.json.

Usage (from the repository root):  python3 tools/verify_p2_archive.py
"""
import json
import sys
from pathlib import Path

PUBLISHED_PREREG_SHA256 = "c870ecb85f69a7e9822c9c48a22ad37c3c21ebb5ceb2ab9a5d46ca42b71a37c8"
P2 = Path(__file__).resolve().parents[1] / "p2"
sys.path.insert(0, str(P2))

from p2.analyze import run_analysis                                    # noqa: E402
from p2.banks import load as load_banks                                # noqa: E402
from p2.integrity import frozen_code_sha, sha_file, verify_frozen      # noqa: E402
from p2.run_full64 import select_attempt, verify_complete_run          # noqa: E402


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    check(sha_file(P2 / "prereg/prereg.json") == PUBLISHED_PREREG_SHA256, "pre-registration hash")
    prereg, psha = verify_frozen(P2 / "prereg/prereg.json", P2 / "banks", root=P2)
    check(True, f"code, configs, prior record and 7 banks match the frozen plan (code {frozen_code_sha(P2)[:12]})")
    banks = load_banks(P2 / "banks")
    items = banks["g3_boundary_test"] + banks["g3_utility_test"]
    weights = {}
    for line in (P2 / "adapter_hashes.txt").read_text().split("\n"):
        if line.strip():
            h, path = line.split(maxsplit=1)
            weights[path.strip().lstrip("./")] = h
    for lib_name in prereg["libraries"]:
        run = P2 / "results" / f"{lib_name}_full64"
        man = verify_complete_run(run, items, psha)
        check(True, f"{lib_name}: 64 output files match ledger and complete.json; records re-validate")
        chosen, info = select_attempt(P2 / "runs" / lib_name)
        lib = chosen
        check(man["selection"] == info and Path(man["library"]).name == lib.name,
              f"{lib_name}: evaluated attempt is the rule's choice ({lib.name}, {info['rule']})")
        check(man["library_json_sha256"] == sha_file(lib / "library.json")
              and man["ledger_sha256"] == sha_file(lib / "ledger.json")
              and man["gate_sha256"] == sha_file(lib / "gate.json")
              and man["prereg_sha256"] == psha and man["code_sha256"] == frozen_code_sha(P2),
              f"{lib_name}: manifest digests match library.json, ledger, gate, prereg and code")
        cfg = json.loads((lib / "library.json").read_text())
        ledger = json.loads((lib / "ledger.json").read_text())
        check(all(weights.get(f"runs/{lib_name}/{lib.name}/adapters/adapter_{r['index']}_{r['name']}.safetensors")
                  == r["sha256"] for r in ledger),
              f"{lib_name}: adapter_hashes.txt matches the training ledger")
        res = run_analysis(run, banks, cfg, prereg)
        stored = json.loads((P2 / "analysis" / lib_name / "results.json").read_text())
        check(res["hypotheses"] == stored["hypotheses"] and res["pair"]["req_leaks"] == stored["pair"]["req_leaks"],
              f"{lib_name}: re-analysis reproduces the stored hypotheses {res['hypotheses']}")
    print("P2 ARCHIVE VERIFIED" if ok else "P2 ARCHIVE VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
