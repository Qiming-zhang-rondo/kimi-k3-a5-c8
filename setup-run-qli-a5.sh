#!/usr/bin/env bash
# Run pinned source QLI DSL vs installed native QLI V2 on Ascend A5.
set -euo pipefail
payload_commit=2fe19671eccf178ec1a0565c1f5ac981800f982a
base_url="https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/$payload_commit/qli-a5"
curl_options=(--fail --silent --show-error --location --retry 2 --connect-timeout 15 --max-time 120)
python_bin=python
work_root=/mnt
download_only=0
test_args=()
while (($#)); do
    case "$1" in
        --insecure) curl_options+=(--insecure); shift ;;
        --proxy) curl_options+=(--proxy "${2:?Missing proxy URL}"); shift 2 ;;
        --work-dir) work_root="${2:?Missing working directory}"; shift 2 ;;
        --python) python_bin="${2:?Missing Python executable}"; shift 2 ;;
        --download-only) download_only=1; shift ;;
        --help|-h)
            echo 'Usage: bash setup-run-qli-a5.sh [--insecure] [--proxy URL] [--work-dir /mnt] [--python python] [--download-only] [-- TEST_ARGS...]'
            echo 'Creates a fresh venv inheriting the current CANN/PyTorch environment, installs only cannbot-dsl==0.7.0 in it, then runs the comparison.'
            echo 'Default suite: smoke. Full comparison: append -- --suite full --profile'
            echo 'No CANN/driver upgrade. Source DSL is explicitly selected instead of the unavailable ds41 package.'
            exit 0 ;;
        --) shift; test_args+=("$@"); break ;;
        --dsl-source|--dsl-root) echo 'This launcher uses its pinned DSL source; use run-qli-a5.sh for custom source selection.' >&2; exit 2 ;;
        *) test_args+=("$1"); shift ;;
    esac
done
command -v curl >/dev/null || { echo 'curl is required.' >&2; exit 1; }
if (( ! download_only )); then
    "$python_bin" - <<'PY'
import platform, sys
if platform.system() != 'Linux' or sys.version_info[:2] not in ((3,10),(3,11),(3,12)):
    raise SystemExit('Requires Linux CPython 3.10/3.11/3.12; select the working A5 Python environment with --python.')
import torch, torch_npu, numpy
if not hasattr(torch, 'float4_e2m1fn_x2') or not hasattr(torch, 'float8_e8m0fnu'):
    raise SystemExit('This Python environment lacks native MXFP4/E8M0 dtypes.')
print('Base Python:', sys.executable)
print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
PY
fi
mkdir -p "$work_root"
run_dir=$(mktemp -d "$work_root/qli-a5-ready.XXXXXX")
echo "Working directory: $run_dir"
echo "Pinned payload: $payload_commit"
verify() {
    local file="$1" expected="$2" actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        echo 'sha256sum or shasum is required.' >&2; exit 1
    fi
    [[ "$actual" == "$expected" ]] || { echo "Checksum mismatch: $file" >&2; exit 1; }
}
fetch() {
    local rel="$1" expected="$2"
    mkdir -p "$(dirname "$run_dir/$rel")"
    curl "${curl_options[@]}" "$base_url/$rel" -o "$run_dir/$rel"
    verify "$run_dir/$rel" "$expected"
}
fetch test_qli_a5.py ae03ff4c3f24f9358afdc384be5c687c9c6bfd7589a850163807dc01ca12a234
fetch QLI_A5_README.txt 74d09ab96664c77ac7ee37a4b6c89dfde7dc26d9cf82a7df088d6e39e11038eb
fetch dsl-wheels.json 42b84ddde5ca1fdc346beb148c526000943fa21c9d9c146f1a2687956608400e
fetch cannbot-source/LICENSE 7a2c992300c275afed721f5b69e950a2382b1b8f8f3f08e1858d7ee4d3134365
fetch cannbot-source/samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py e5152d247b0ddd26a186b42a0a40d3ecc5a6f026d9fd2b0f43a96e41df1e00e6
fetch cannbot-source/samples/quant_lightning_indexer_metadata_dsl/quant_lightning_indexer_metadata_dsl.py f432d48d24a2a0adf600bd6baf6b014ecb4fe319d52531128e6ff6d8e0ea8f38
echo 'Test and unmodified upstream source checksums verified.'
if (( download_only )); then
    echo "Download complete: $run_dir"; exit 0
fi
wheel_spec=$("$python_bin" - "$run_dir/dsl-wheels.json" <<'PY'
import json, platform, sys
data = json.load(open(sys.argv[1]))
arch = platform.machine().lower()
arch = {'amd64':'x86_64','arm64':'aarch64'}.get(arch, arch)
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
name = f'cannbot_dsl-0.7.0-{tag}-{tag}-manylinux_2_28_{arch}.whl'
for item in data['wheels']:
    if item['filename'] == name:
        print('\t'.join((name,item['url'],item['sha256'])))
        break
else:
    raise SystemExit(f'No pinned wheel for {tag} / {arch}')
PY
)
IFS=$'\t' read -r wheel_file wheel_url wheel_sha <<< "$wheel_spec"
echo "Downloading official wheel: $wheel_file"
curl "${curl_options[@]}" "$wheel_url" -o "$run_dir/$wheel_file"
verify "$run_dir/$wheel_file" "$wheel_sha"
"$python_bin" -m venv --system-site-packages "$run_dir/venv"
venv_python="$run_dir/venv/bin/python"
# Ignore inherited versions only for this one package; no dependency resolver or index access.
"$venv_python" -m pip install --no-index --no-deps --ignore-installed "$run_dir/$wheel_file" 2>&1 | tee "$run_dir/dsl-install.log"
"$venv_python" - "$run_dir/cannbot-source" <<'PY' 2>&1 | tee "$run_dir/dsl-preflight.log"
import importlib.metadata, pathlib, sys
import torch, torch_npu, cannbotdsl
from cannbotdsl.aicpu import aicpu_kernel
from cannbotdsl.lang.host import host
print('DSL version:', importlib.metadata.version('cannbot-dsl'))
print('DSL location:', cannbotdsl.__file__)
root = pathlib.Path(sys.argv[1])/'samples'
sys.path[:0] = [str(root/'quant_lightning_indexer_dsl'), str(root/'quant_lightning_indexer_metadata_dsl')]
import quant_lightning_indexer_dsl
import quant_lightning_indexer_metadata_dsl
print('PASS: QLI DSL source and metadata imports. NPU compilation/runtime still require the following tests.')
PY
cd "$run_dir"
"$venv_python" test_qli_a5.py --self-test
echo "Running native QLI V2 versus source QLI DSL. Results: $run_dir"
echo "Rerun without downloading: $venv_python $run_dir/test_qli_a5.py --dsl-source samples --dsl-root $run_dir/cannbot-source --suite full --profile"
"$venv_python" test_qli_a5.py "${test_args[@]}" --dsl-source samples --dsl-root "$run_dir/cannbot-source" 2>&1 | tee "$run_dir/test.log"
