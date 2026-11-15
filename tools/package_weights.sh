#!/usr/bin/env bash
# Package the P1 and P2 adapter weights for the Zenodo record https://doi.org/10.5281/zenodo.24261411.
#
# Run from the repository root on the machine that holds the trained adapters:
#   bash tools/package_weights.sh <P1_ROOT> <P2_ROOT> [OUT_DIR]
# P1_ROOT / P2_ROOT are the directories that contain runs/pilot_lib1 and runs/L2, runs/L3
# (the paths in p1/adapter_hashes.txt and p2/adapter_hashes.txt are relative to them).
# Every adapter is checked against the archived training hashes first; nothing is packaged
# if a single file is missing or differs.
set -euo pipefail
P1_ROOT=$(realpath "${1:?P1_ROOT missing}")
P2_ROOT=$(realpath "${2:?P2_ROOT missing}")
REPO=$(pwd)
OUT=$(realpath -m "${3:-$REPO/release_assets}")
[ -f "$REPO/p1/adapter_hashes.txt" ] && [ -f "$REPO/p2/adapter_hashes.txt" ] || {
  echo "run this from the CoLoSS repository root" >&2; exit 1; }
mkdir -p "$OUT"

echo "== verifying adapters against the archived hashes"
(cd "$P1_ROOT" && sha256sum -c --strict "$REPO/p1/adapter_hashes.txt")
(cd "$P2_ROOT" && sha256sum -c --strict "$REPO/p2/adapter_hashes.txt")

pack () {  # pack <root> <zip name> <dir>...
  local root=$1 name=$2; shift 2
  local list; list=$(mktemp)
  for d in "$@"; do
    [ -d "$root/$d" ] || { echo "missing $root/$d" >&2; exit 1; }
    (cd "$root" && find "$d" -maxdepth 2 -type f \
      \( -name '*.safetensors' -o -name 'library.json' -o -name 'ledger.json' -o -name 'train_log.jsonl' \)) >> "$list"
  done
  sort -o "$list" "$list"
  rm -f "$OUT/$name"
  (cd "$root" && zip -X -q "$OUT/$name" -@ < "$list")
  echo "   $name: $(wc -l < "$list") files"
  rm -f "$list"
}

echo "== building release assets in $OUT"
pack "$P1_ROOT" CoLoSS_P1_Weights.zip runs/pilot_lib1
pack "$P2_ROOT" CoLoSS_P2_L2_Weights.zip runs/L2/attempt_1 runs/L2/attempt_2 runs/L2/attempt_3
pack "$P2_ROOT" CoLoSS_P2_L3_Weights.zip runs/L3/attempt_1 runs/L3/attempt_2 runs/L3/attempt_3
(cd "$OUT" && sha256sum CoLoSS_P1_Weights.zip CoLoSS_P2_L2_Weights.zip CoLoSS_P2_L3_Weights.zip > CoLoSS_Weight_Archives_SHA256SUMS.txt)

echo "== done; asset checksums:"
cat "$OUT/CoLoSS_Weight_Archives_SHA256SUMS.txt"
du -h "$OUT"/*.zip
echo
echo "Upload these files to the Zenodo record https://doi.org/10.5281/zenodo.24261411:"
echo "  $OUT/CoLoSS_P*_Weights.zip $OUT/CoLoSS_Weight_Archives_SHA256SUMS.txt"
