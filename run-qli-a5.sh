#!/usr/bin/env bash
# Download a pinned QLI test bundle, verify it, and run in the current A5 environment.
set -euo pipefail
payload_commit=5480842f5f886ea95d400351bce6ac56d15cea85
base_url="https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/${payload_commit}/qli-a5"
curl_options=(--fail --silent --show-error --location --retry 2 --connect-timeout 15 --max-time 120)
work_root=/mnt
python_bin=python
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
            echo 'Usage: bash run-qli-a5.sh [--insecure] [--proxy URL] [--work-dir /mnt] [--python python] [--download-only] [-- TEST_ARGS...]'
            echo 'Default: download into a fresh directory, run CPU self-test, then NPU smoke tests.'
            echo 'Example: bash run-qli-a5.sh --insecure -- --suite full --profile'
            echo 'Use the existing CANN/PyTorch/operator environment; no packages are installed.'
            exit 0 ;;
        --) shift; test_args+=("$@"); break ;;
        *) test_args+=("$1"); shift ;;
    esac
done
command -v curl >/dev/null || { echo 'curl is required.' >&2; exit 1; }
if (( ! download_only )); then
    command -v "$python_bin" >/dev/null || { echo 'Python not found; specify --python /path/to/python.' >&2; exit 1; }
fi
mkdir -p "$work_root"
download_dir=$(mktemp -d "$work_root/qli-a5.XXXXXX")
echo "Pinned source: $payload_commit"
echo "Working directory: $download_dir"
check_file() {
    local file="$1" expected="$2" actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        echo 'sha256sum or shasum is required.' >&2; exit 1
    fi
    [[ "$actual" == "$expected" ]] || { echo "Checksum mismatch: $file; no test executed." >&2; exit 1; }
}
curl "${curl_options[@]}" "$base_url/test_qli_a5.py" -o "$download_dir/test_qli_a5.py"
check_file "$download_dir/test_qli_a5.py" ae03ff4c3f24f9358afdc384be5c687c9c6bfd7589a850163807dc01ca12a234
curl "${curl_options[@]}" "$base_url/QLI_A5_README.txt" -o "$download_dir/QLI_A5_README.txt"
check_file "$download_dir/QLI_A5_README.txt" 74d09ab96664c77ac7ee37a4b6c89dfde7dc26d9cf82a7df088d6e39e11038eb
echo 'Both file checksums verified.'
if (( download_only )); then
    echo "Download complete. Instructions: $download_dir/QLI_A5_README.txt"
    exit 0
fi
cd "$download_dir"
"$python_bin" test_qli_a5.py --self-test
echo 'Starting NPU tests; source your CANN environment and select your A5 Python environment first.'
echo "Results will be under $download_dir unless --output specifies another path."
exec "$python_bin" test_qli_a5.py "${test_args[@]}"
