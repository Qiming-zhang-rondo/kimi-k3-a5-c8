#!/usr/bin/env bash
set -euo pipefail
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
va_source=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --insecure) curl_options+=(--insecure); shift ;;
        --proxy) curl_options+=(--proxy "${2:?Missing proxy URL}"); shift 2 ;;
        --source) va_source="${2:?Missing VA source path}"; shift 2 ;;
        *) echo 'Usage: bash apply-c8-public-switch.sh [--insecure] [--proxy URL] [--source VA_DIR]' >&2; exit 2 ;;
    esac
done
if [[ -z "$va_source" ]]; then
    va_source=$(python -c 'import importlib.util,pathlib; s=importlib.util.find_spec("vllm_ascend"); assert s and s.origin; print(pathlib.Path(s.origin).resolve().parent.parent)')
fi
cd "$va_source"
[[ -f vllm_ascend/attention/kimi_k3_sfa_c8_graph.py ]] || { echo 'Install the complete K3 C8 patch first. No changes made.' >&2; exit 1; }
git rev-parse --show-toplevel >/dev/null
download_dir=$(mktemp -d /tmp/k3-c8-public-switch.XXXXXX)
patch_file="$download_dir/k3-c8-public-switch-va029.patch"
curl "${curl_options[@]}" https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/58521f9dea81f736f352c30c91ece627e0de6301/k3-c8-public-switch-va029.patch --output "$patch_file"
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$patch_file" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$patch_file" | awk '{print $1}')
fi
[[ "$actual" == b770b5a8da91cf74b48061332e52134ab522cc0d80c523901dd3038fd969cbc4 ]] || { echo 'SHA256 mismatch; no changes made.' >&2; exit 1; }
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo 'K3 C8 public-switch patch already applied.'
elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo 'Applied K3 C8 public-switch patch.'
else
    echo 'Source does not match the VA 0.29 C8 baseline; no changes made. Do not force apply.' >&2
    exit 1
fi
echo "VA source: $PWD"
echo 'Use enable_sparse_sfa_c8=true with kv-cache-dtype auto. Legacy K3 option remains compatible.'
echo 'Restart the model service to load the change. No NPU tests or service restarts were performed.'
