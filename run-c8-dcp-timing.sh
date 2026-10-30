#!/usr/bin/env bash
# Fetch immutable diagnostic files only; never apply a framework patch.
set -euo pipefail
K3_DCP_CURL=(curl --fail --silent --show-error --location --connect-timeout 15 --max-time 120)
K3_DCP_ARGS=()
while (($#)); do
    case "$1" in
        --insecure) K3_DCP_CURL+=(--insecure); shift ;;
        --proxy) K3_DCP_CURL+=(--proxy "${2:?Provide proxy URL}"); shift 2 ;;
        *) K3_DCP_ARGS+=("$1"); shift ;;
    esac
done

K3_DCP_DIR=$(mktemp -d /mnt/c8-dcp-src.XXXXXX)
K3_DCP_BASE=https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8
"${K3_DCP_CURL[@]}" "$K3_DCP_BASE/573d0184d52ff17bb11742858014c0ddb952195d/c8-empty.py" -o "$K3_DCP_DIR/c8-empty.py"
"${K3_DCP_CURL[@]}" "$K3_DCP_BASE/57d6d2e75c485c6c1492809d3476bc2e7c6f4ccf/c8-dcp-timing.py" -o "$K3_DCP_DIR/c8-dcp.py"
python - "$K3_DCP_DIR" <<'PY'
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "c8-empty.py": "4b5fe2982ebeaa8e12c74740b85b4c9b87977650a0dbd200ceeddc08ed55300b",
    "c8-dcp.py": "9f6cfe01fbdd2dfc7771c074dbacd44caf2d473213f255e427d0f801ea6c3e32",
}
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Checksum mismatch: {name}; NPU workers were not started")
print(f"Verified diagnostic files: {root}", flush=True)
PY
exec python "$K3_DCP_DIR/c8-dcp.py" "${K3_DCP_ARGS[@]}"
