#!/usr/bin/env bash
# Apply the targeted repair, then run one-card native address/isolation tests.
set -euo pipefail

K3_FIX_REF=53c466bff94eaced28396d29d22d7100ecbac1a2
K3_FIX_BASE="https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/${K3_FIX_REF}"
K3_FIX_CARD=""
K3_FIX_INSTALL_ONLY=0
K3_FIX_CURL=(curl --fail --silent --show-error --location --connect-timeout 15 --max-time 120)
K3_FIX_ARGS=()
while (($#)); do
    case "$1" in
        --card) K3_FIX_CARD="${2:?Provide one authorized physical A5 card ID}"; shift 2 ;;
        --insecure) K3_FIX_CURL+=(--insecure); shift ;;
        --proxy) K3_FIX_CURL+=(--proxy "${2:?Provide proxy URL}"); shift 2 ;;
        --va-source) K3_FIX_ARGS+=(--va-source "${2:?Provide VA source root}"); shift 2 ;;
        --install-only) K3_FIX_INSTALL_ONLY=1; shift ;;
        *) echo "Usage: bash repair-c8.sh --card N [--insecure] [--proxy URL] [--va-source DIR] [--install-only]" >&2; exit 2 ;;
    esac
done
if [[ "$K3_FIX_INSTALL_ONLY" == 0 && ! "$K3_FIX_CARD" =~ ^[0-9]+$ ]]; then
    echo "Choose one authorized A5 with --card N, or use --install-only. No changes made." >&2
    exit 2
fi

K3_FIX_TMP=$(mktemp -d -t k3-c8-repair.XXXXXX)
trap 'rm -f -- "$K3_FIX_TMP/install.py" "$K3_FIX_TMP/diag.py"; rmdir -- "$K3_FIX_TMP"' EXIT
"${K3_FIX_CURL[@]}" "${K3_FIX_BASE}/install-c8-cache-isolation.py" -o "$K3_FIX_TMP/install.py"
"${K3_FIX_CURL[@]}" "${K3_FIX_BASE}/diagnose-k3-c8-cache-addressing.py" -o "$K3_FIX_TMP/diag.py"
python - "$K3_FIX_TMP" <<'PY'
import hashlib
import sys
from pathlib import Path
expected = {
    "install.py": "8c5c81fbd70cbc70e83c8d1d62044e66cd81e08e7e5cfd860de050731f01c2b5",
    "diag.py": "a3cc60fe604b975c187a18178d2057d53fd8a86dacfdbd87ad1f758ec97c980f",
}
for name, digest in expected.items():
    if hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Download checksum mismatch: {name}; no patch applied")
PY
python "$K3_FIX_TMP/install.py" "${K3_FIX_ARGS[@]}"
if [[ "$K3_FIX_INSTALL_ONLY" == 1 ]]; then
    exit 0
fi
K3_FIX_LOG=$(mktemp /mnt/c8-cache-diag.XXXXXX.log)
echo "Testing one physical card: ${K3_FIX_CARD}. Log: ${K3_FIX_LOG}"
K3_FIX_DIAG_ARGS=()
if ((${#K3_FIX_ARGS[@]})); then
    K3_FIX_DIAG_ARGS+=(--va-root "${K3_FIX_ARGS[1]}")
fi
ASCEND_RT_VISIBLE_DEVICES="$K3_FIX_CARD" python "$K3_FIX_TMP/diag.py" "${K3_FIX_DIAG_ARGS[@]}" 2>&1 | tee "$K3_FIX_LOG"
