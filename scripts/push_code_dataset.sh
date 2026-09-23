#!/usr/bin/env bash
# Publishes the repo's Python code as the private Kaggle Dataset `knee-code`, which the offline
# submission notebook imports (it cannot git clone with internet OFF).
# Requires the Kaggle CLI with credentials (~/.kaggle/kaggle.json).
#
#   KAGGLE_USER=<your-username> bash scripts/push_code_dataset.sh "message"
set -euo pipefail

: "${KAGGLE_USER:?set KAGGLE_USER to the Kaggle account that owns the dataset}"
MSG="${1:-update $(git rev-parse --short HEAD 2>/dev/null || date +%F)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"

rsync -a --prune-empty-dirs \
    --exclude='tests/' --exclude='kaggle/' --exclude='scripts/' --exclude='.git/' \
    --include='*/' --include='*.py' --include='requirements.txt' --exclude='*' \
    "$ROOT/" "$STAGE/"

cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "knee-code", "id": "${KAGGLE_USER}/knee-code", "licenses": [{"name": "CC0-1.0"}]}
EOF

if kaggle datasets status "${KAGGLE_USER}/knee-code" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE" -m "$MSG" --dir-mode zip
else
    kaggle datasets create -p "$STAGE" --dir-mode zip
fi
rm -rf "$STAGE"
