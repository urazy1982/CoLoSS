#!/usr/bin/env bash
# Pack everything needed for the paper EXCEPT adapter weights (shared only on request).
set -euo pipefail
find runs -name '*.safetensors' -print0 | sort -z | xargs -0 sha256sum > adapter_hashes.txt
OUT="H1_P2_results_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar --exclude='*.safetensors' -czf "$OUT" prereg banks/bank_hashes.json logs adapter_hashes.txt \
    $(ls -d runs results analysis 2>/dev/null)
sha256sum "$OUT" | tee "$OUT.sha256"
ls -lh "$OUT"
