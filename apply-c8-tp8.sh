#!/usr/bin/env bash
# Follow-up to the VA 0.29 C8 + scatter + neutral-sink patches.
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 90)
run_tests=true
for option in "$@"; do
    case "$option" in
        --insecure) curl_options+=(--insecure) ;;
        --apply-only) run_tests=false ;;
        *) echo "Usage: bash apply-c8-tp8.sh [--insecure] [--apply-only] (from VA source directory)" >&2; exit 2 ;;
    esac
done
if [[ ! -f vllm_ascend/attention/kimi_k3_sfa_c8.py ]]; then
    echo "Run from the VA 0.29 source directory with the previous C8 fixes installed." >&2
    exit 1
fi
git rev-parse --show-toplevel >/dev/null
download_dir=$(mktemp -d /tmp/k3-c8-tp8.XXXXXX)
patch_file="$download_dir/k3-c8-tp8-head-padding-va029.patch"
curl "${curl_options[@]}" https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/10b1ae3a6fe53f2cff1d9be3d6410bcfc9393b1b/k3-c8-tp8-head-padding-va029.patch --output "$patch_file"
expected_sha=c392b89920d6f2764c1417e1c98d215eaa80d8ac630dfa19fdcbdfa87490b31b
if command -v sha256sum >/dev/null 2>&1; then
    actual_sha=$(sha256sum "$patch_file" | awk '{print $1}')
else
    actual_sha=$(shasum -a 256 "$patch_file" | awk '{print $1}')
fi
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Patch SHA256 mismatch; source was not modified." >&2
    exit 1
fi
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "TP8 head-padding patch already applied."
elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo "Applied TP8 head padding. Restart the model process before testing it."
else
    echo "Patch conflict; source was not overwritten. This patch requires the earlier scatter and neutral-sink fixes." >&2
    exit 1
fi
if [[ "$run_tests" == false ]]; then
    echo "Applied only; no NPU tests were run."
    exit 0
fi
log_file="$download_dir/native-tests.log"
echo "Running all 20 native cases; use idle NPU devices. Log: $log_file"
VLLM_PLUGINS=ascend python -m pytest -sv --noconftest --tb=short \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8.py \
    2>&1 | tee "$log_file"
