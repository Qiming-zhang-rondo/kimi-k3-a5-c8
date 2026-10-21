#!/usr/bin/env bash
# VA 0.29: add FULL_DECODE_ONLY to the previously installed native K3 C8 feature.
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
run_tests=true
for option in "$@"; do
    case "$option" in
        --insecure) curl_options+=(--insecure) ;;
        --apply-only) run_tests=false ;;
        *) echo "Usage: bash apply-c8-graph.sh [--insecure] [--apply-only]" >&2; exit 2 ;;
    esac
done
if [[ ! -f vllm_ascend/attention/kimi_k3_sfa_c8.py ]]; then
    va_source=$(python -c 'import importlib.util,pathlib; s=importlib.util.find_spec("vllm_ascend"); assert s and s.origin; print(pathlib.Path(s.origin).resolve().parent.parent)')
    cd "$va_source"
fi
if [[ ! -f vllm_ascend/attention/kimi_k3_sfa_c8.py ]]; then
    echo "Previous VA 0.29 K3 C8 + scatter/sink fixes are required. No source changed." >&2
    exit 1
fi
git rev-parse --show-toplevel >/dev/null
echo "VA source: $PWD"
download_dir=$(mktemp -d /tmp/k3-c8-graph.XXXXXX)
echo "Downloads and complete test logs: $download_dir"

fetch_checked() {
    local name=$1 revision=$2 expected=$3 actual
    curl "${curl_options[@]}" "https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/$revision/$name" --output "$download_dir/$name"
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$download_dir/$name" | awk '{print $1}')
    else
        actual=$(shasum -a 256 "$download_dir/$name" | awk '{print $1}')
    fi
    [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch for $name; aborting." >&2; exit 1; }
}

fetch_checked k3-c8-full-decode-graph-va029.patch 70b389f1cedf9feb95c0caf420ad23adeec72669 d12dfa45004e1d0c42b7977a9656d5b5c1aa2bfabdf1508649fe28240c3a8a61
graph_patch="$download_dir/k3-c8-full-decode-graph-va029.patch"
if git apply --reverse --check "$graph_patch" 2>/dev/null; then
    echo "This exact graph patch is already applied."
elif git apply --check "$graph_patch" 2>/dev/null; then
    git apply "$graph_patch"
    echo "Applied graph support on top of the TP8 patch."
else
    # Also accept the earlier 10-pass version, before TP head-padding.
    # Check one cumulative patch atomically; never half-apply a patch chain.
    fetch_checked k3-c8-tp-and-graph-va029.patch bb3da03479525f75b140faa10d34a9f3e114982e d134a0290f88ff29b286f61de6c57d18e52297006eff648a22424ae3ca91b8ae
    cumulative_patch="$download_dir/k3-c8-tp-and-graph-va029.patch"
    if git apply --check "$cumulative_patch"; then
        git apply "$cumulative_patch"
        echo "Applied TP head padding and graph support together."
    else
        echo "Source differs from both supported baselines. Nothing was overwritten; keep the conflict output." >&2
        exit 1
    fi
fi
echo "Restart existing model processes to use the updated Python source. No dependency reinstall is required."
if [[ "$run_tests" == false ]]; then
    echo "Apply-only: no NPU test has been performed."
    exit 0
fi

python -c 'import torch, torch_npu; assert torch.npu.is_available(), "A real idle NPU is required"'
echo "Running all 34 native cases (20 eager + 14 graph). No fail-fast; use an idle NPU."
VLLM_PLUGINS=ascend python -m pytest -sv --noconftest --tb=short \
    --junitxml="$download_dir/native-tests.xml" \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8.py \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8_graph.py \
    2>&1 | tee "$download_dir/native-tests.log"
python - "$download_dir/native-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall(".//testcase")
assert len(cases) == 34, f"Expected 34 native cases, got {len(cases)}"
assert all(not any(case.find(tag) is not None for tag in ("skipped", "failure", "error")) for case in cases), "Skipped/failed native cases are not an A5 pass"
print("All 34 native cases passed. Next: restart the reduced model and test actual FULL_DECODE_ONLY replay.")
PY
