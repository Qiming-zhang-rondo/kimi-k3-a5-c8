#!/usr/bin/env bash
# Follow-up to apply-c8-graph.sh; changes tests only, not model code/weights.
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
run_tests=true
for option in "$@"; do
    case "$option" in
        --insecure) curl_options+=(--insecure) ;;
        --apply-only) run_tests=false ;;
        *) echo 'Usage: bash apply-c8-graph-stream-fix.sh [--insecure] [--apply-only]' >&2; exit 2 ;;
    esac
done
test_file=tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_sfa_c8_graph.py
if [[ ! -f "$test_file" ]]; then
    va_source=$(python -c 'import importlib.util,pathlib; s=importlib.util.find_spec("vllm_ascend"); assert s and s.origin; print(pathlib.Path(s.origin).resolve().parent.parent)')
    cd "$va_source"
fi
[[ -f "$test_file" ]] || { echo 'Previous K3 C8 graph patch is required; no source changed.' >&2; exit 1; }
git rev-parse --show-toplevel >/dev/null
download_dir=$(mktemp -d /tmp/k3-c8-stream-fix.XXXXXX)
echo "VA source: $PWD; patch and logs: $download_dir"
patch_file="$download_dir/fix-c8-graph-capture-stream-va029.patch"
curl "${curl_options[@]}" https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/39399fc48811ae35e3639a69c27d29c8aa875d16/fix-c8-graph-capture-stream-va029.patch --output "$patch_file"
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$patch_file" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$patch_file" | awk '{print $1}')
fi
[[ "$actual" == ea1720f24f20fc78249b14ed14a8f4c135a6f1050686a43c6d2336c76bc62a97 ]] || { echo 'SHA256 mismatch; no source changed.' >&2; exit 1; }
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'Capture-stream fix already applied.'
elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo 'Applied test-only capture-stream fix.'
else
    echo 'Patch conflict; source was not overwritten.' >&2
    exit 1
fi
if [[ "$run_tests" == false ]]; then
    echo 'Apply-only: no NPU validation performed.'
    exit 0
fi
echo 'Rerunning all 14 graph cases on an idle A5. No model service will be started.'
VLLM_PLUGINS=ascend python -m pytest -sv --noconftest --tb=short \
    --junitxml="$download_dir/graph-tests.xml" "$test_file" 2>&1 | tee "$download_dir/graph-tests.log"
python - "$download_dir/graph-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases) == 14, f'Expected 14 graph cases, got {len(cases)}'
assert all(not any(c.find(tag) is not None for tag in ('skipped', 'failure', 'error')) for c in cases), 'Skipped or failed cases are not an A5 pass'
print('All 14 native graph cases passed. Model-level validation remains separate.')
PY
