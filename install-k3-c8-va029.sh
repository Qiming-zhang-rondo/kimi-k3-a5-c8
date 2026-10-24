#!/usr/bin/env bash
# Complete K3 C8 feature for a fresh, compatible VA 0.29 editable source tree.
# No dependency install, server launch, global TLS change, or forced patching.
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
source_dir=''
run_tests=false
while (($#)); do
    case "$1" in
        --insecure) curl_options+=(--insecure); shift ;;
        --proxy) [[ $# -ge 2 ]] || exit 2; curl_options+=(--proxy "$2"); shift 2 ;;
        --source) [[ $# -ge 2 ]] || exit 2; source_dir=$2; shift 2 ;;
        --test) run_tests=true; shift ;;
        *) echo 'Usage: bash install-k3-c8-va029.sh [--proxy URL] [--insecure] [--test] [--source VA_DIR]' >&2; exit 2 ;;
    esac
done
if [[ -z "$source_dir" ]]; then
    source_dir=$(python -c 'import importlib.util,importlib.metadata,pathlib; v=importlib.metadata.version("vllm-ascend"); assert v.startswith("0.29."), f"Expected VA 0.29, installed {v}"; s=importlib.util.find_spec("vllm_ascend"); assert s and s.origin; print(pathlib.Path(s.origin).resolve().parent.parent)')
fi
cd "$source_dir"
[[ -f vllm_ascend/attention/mla_v1.py && -f vllm_ascend/worker/v2/attn_utils.py ]] || {
    echo 'Not a compatible VA source directory; nothing changed.' >&2; exit 1;
}
repo_root=$(git rev-parse --show-toplevel)
[[ "$(cd "$repo_root" && pwd -P)" == "$(pwd -P)" ]] || {
    echo 'VA package is not at its Git repository root; nothing changed.' >&2; exit 1;
}
echo "Target VA source: $PWD"
echo "Current HEAD: $(git rev-parse HEAD)"
echo 'Patch baseline: VA 0.29 b3d3bb6805ffaf8fcc20086d5e59dbf2352083b4; other revisions must pass git apply --check.'
download_dir=$(mktemp -d /tmp/k3-c8-full.XXXXXX)
patch_file="$download_dir/k3-c8-va029-complete-20261024.patch"
echo "Retained patch and logs: $download_dir"
curl "${curl_options[@]}" https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/eaaa6aa683b8e32df542b5e375b8338fd2aa3b81/k3-c8-va029-complete-20261024.patch --output "$patch_file"
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$patch_file" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$patch_file" | awk '{print $1}')
fi
[[ "$actual" == ee7bb9127970f6f8e26cffc72081b9c990ca54a1ea1f39c236b6308bc4b8b10a ]] || {
    echo 'Patch SHA256 mismatch; source was not modified.' >&2; exit 1;
}
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'Complete C8 feature is already applied.'
elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo 'Applied complete C8 feature, including the non-default capture-stream fix.'
else
    echo 'Baseline differs or earlier partial patches exist. No changes were applied; send the conflict output. Do not force or reset the repository.' >&2
    exit 1
fi
echo 'No dependency reinstall or weight conversion performed. Restart existing model processes before using the change.'
if [[ "$run_tests" == false ]]; then
    echo 'Patch installation finished. No NPU tests or model services were started.'
    echo 'Validation from the old image does not certify this new image. Rerun this installer with --test on an idle A5.'
    exit 0
fi
python -c 'import torch,torch_npu; assert torch.npu.is_available(), "A real idle NPU is required"'
echo 'Running 34 native cases (20 eager + 14 graph); no model service will be launched.'
VLLM_PLUGINS=ascend python -m pytest -sv --noconftest --tb=short \
    --junitxml="$download_dir/native-tests.xml" \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8.py \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8_graph.py \
    2>&1 | tee "$download_dir/native-tests.log"
python - "$download_dir/native-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases) == 34, f'Expected 34 native cases, got {len(cases)}'
assert all(not any(c.find(tag) is not None for tag in ('skipped', 'failure', 'error')) for c in cases), 'Skipped/failed cases are not an A5 pass'
print('All 34 native cases passed on this image. Model validation remains separate.')
PY
