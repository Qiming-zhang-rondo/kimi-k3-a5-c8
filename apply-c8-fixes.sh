#!/usr/bin/env bash
# Apply reviewed follow-up fixes to the already-installed VA 0.29 C8 patch.
set -euo pipefail

repo_dir="${1:-$PWD}"
cd "$repo_dir"
if [[ ! -f vllm_ascend/attention/kimi_k3_sfa_c8.py ]]; then
    echo "Run this command from the VA 0.29 source directory after applying the initial C8 patch." >&2
    exit 1
fi
git rev-parse --show-toplevel >/dev/null
download_dir=$(mktemp -d /tmp/k3-c8-fixes.XXXXXX)
patch_file="$download_dir/fix-scatter-keyword-va029.patch"
curl --fail --location --retry 2 --connect-timeout 15 --max-time 90 \
    https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/e3d76227e83e489071e1256e4f56670fc56285ad/fix-scatter-keyword-va029.patch \
    --output "$patch_file"
if git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo "Applied ScatterPaCache keyword-only fix."
elif git apply --reverse --check "$patch_file"; then
    echo "ScatterPaCache fix already applied; leaving source unchanged."
else
    echo "Source differs from the known baseline. No files were overwritten. Patch: $patch_file" >&2
    exit 1
fi

log_file="$download_dir/native-tests.log"
echo "Running all native cases; full log: $log_file"
# Keep every failure, not just the first one. pipefail preserves pytest's status.
VLLM_PLUGINS=ascend python -m pytest -sv --noconftest --tb=short \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8.py \
    2>&1 | tee "$log_file"
