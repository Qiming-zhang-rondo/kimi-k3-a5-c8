#!/usr/bin/env bash
# Apply-only complete K3 C8 installer. No NPU tests, dependency installs or service starts.
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
source_dir=''
while (($#)); do
    case "$1" in
        --insecure) curl_options+=(--insecure); shift ;;
        --proxy) curl_options+=(--proxy "${2:?Missing proxy URL}"); shift 2 ;;
        --source) source_dir="${2:?Missing source directory}"; shift 2 ;;
        *) echo 'Usage: bash install.sh [--insecure] [--proxy URL] [--source VA_DIR]' >&2; exit 2 ;;
    esac
done
if [[ -z "$source_dir" ]]; then
    source_dir=$(python -c 'import importlib.util,importlib.metadata,pathlib; v=importlib.metadata.version("vllm-ascend"); assert v.startswith("0.29."), f"Expected VA 0.29, got {v}"; s=importlib.util.find_spec("vllm_ascend"); assert s and s.origin; print(pathlib.Path(s.origin).resolve().parent.parent)')
fi
cd "$source_dir"
[[ -f vllm_ascend/attention/mla_v1.py && -f vllm_ascend/worker/v2/attn_utils.py ]] || { echo 'Not a compatible VA source tree.' >&2; exit 1; }
repo_root=$(git rev-parse --show-toplevel)
[[ "$(cd "$repo_root" && pwd -P)" == "$(pwd -P)" ]] || { echo 'VA source must be a Git repository root.' >&2; exit 1; }
download_dir=$(mktemp -d /tmp/k3-c8-install.XXXXXX)
patch_file="$download_dir/complete.patch"
curl "${curl_options[@]}" https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/124a8cd22692ffc2e774d184ac85b57ae0b78895/k3-c8-va029-complete-20261028.patch -o "$patch_file"
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$patch_file" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$patch_file" | awk '{print $1}')
fi
[[ "$actual" == 65ebc7aade763b4f650ee9d5484e48db7bfd9f9a1b34a0d002ca83b470f3a942 ]] || { echo 'Checksum mismatch; nothing changed.' >&2; exit 1; }
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'Complete C8 feature and public switch already installed.'
elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo 'Installed complete C8 feature including enable_sparse_sfa_c8 support.'
else
    echo 'Baseline differs or earlier partial C8 patches exist. No changes made; do not force/reset. Send this output for migration.' >&2
    exit 1
fi
echo "VA source: $PWD"
echo 'Use enable_sparse_sfa_c8=true and kv-cache-dtype auto. Restart your model service manually.'
echo 'Apply only: no NPU tests, dependency installation, model launch or automatic restart.'
