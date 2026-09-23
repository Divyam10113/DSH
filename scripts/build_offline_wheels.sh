#!/usr/bin/env bash
# Downloads the wheels the offline submission notebook needs (internet is disabled at scoring time).
# Run INSIDE a Kaggle notebook with internet ON so the wheels match Kaggle's Python/platform,
# then publish the output folder as the `knee-wheels` dataset.
#
#   bash scripts/build_offline_wheels.sh /kaggle/working/wheels
set -euo pipefail

OUT="${1:-wheels}"
mkdir -p "$OUT"
python -m pip download --only-binary=:all: -d "$OUT" \
    pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm

echo "Wheels in $OUT:"
ls -1 "$OUT"
