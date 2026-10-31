#!/usr/bin/env bash
# Download pinned Stage 1 tools. No VA patch, CANN upgrade or global pip install.
set -euo pipefail
QFM_REV=c1b154dba87910e45a3611100cd255d0976a471b
QFM_MANIFEST_SHA=24016cb51ea64c421526dbf9f0fa920eac20445a1fd486e99c0fc4f9988a3536
QFM_BASE="https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/$QFM_REV/qfm_stage1"
QFM_CURL=(curl --fail --silent --show-error --location --connect-timeout 15 --max-time 180)
QFM_BUILD_ARGS=()
QFM_RUN_ARGS=()
QFM_DEPS=/mnt/k3-qfm-stage1-deps
QFM_BUILD=0
QFM_INSPECT=0
while (($#)); do
    case "$1" in
        --insecure) QFM_CURL+=(--insecure); QFM_BUILD_ARGS+=(--insecure); shift ;;
        --proxy) QFM_CURL+=(--proxy "${2:?proxy URL required}"); QFM_BUILD_ARGS+=(--proxy "$2"); shift 2 ;;
        --deps-prefix) QFM_DEPS=${2:?private directory required}; shift 2 ;;
        --build-ops) QFM_BUILD=1; shift ;;
        --cann-env|--cann-version-file|--jobs) QFM_BUILD_ARGS+=("$1" "${2:?value required}"); shift 2 ;;
        --suite) [[ ${2:?suite required} == inspect ]] && QFM_INSPECT=1; QFM_RUN_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help)
            echo "qfm1.sh [--insecure] [--proxy URL] [--deps-prefix PATH] --card PHYSICAL --suite inspect|all|small|graph|dcp|performance"
            echo "qfm1.sh [download options] --build-ops [--cann-env PATH --cann-version-file PATH --jobs N]"
            echo "Build/stage only the two QFM operators in a private prefix; never patch VA or upgrade CANN."
            exit 0 ;;
        *) QFM_RUN_ARGS+=("$1"); shift ;;
    esac
done
QFM_DIR=$(mktemp -d /mnt/qfm-stage1-src.XXXXXX)
for QFM_FILE in manifest.json README.md BUILD.md setup_qfm.sh build_qfm.sh planner.py qfm_probe.py prolog_probe.py run.py test_planner.py test_probe.py test_run.py test_setup.py; do
    "${QFM_CURL[@]}" "$QFM_BASE/$QFM_FILE" -o "$QFM_DIR/$QFM_FILE"
done
python3 - "$QFM_DIR" "$QFM_MANIFEST_SHA" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
if hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit("Manifest checksum mismatch. No build or NPU workers started.")
manifest = json.loads((root / "manifest.json").read_text())
for name, digest in manifest["files"].items():
    if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Checksum mismatch: {name}. No build or NPU workers started.")
print(f"Verified Stage 1 tools: {root}", flush=True)
PY
if ((QFM_BUILD)); then
    exec bash "$QFM_DIR/setup_qfm.sh" --prefix "$QFM_DEPS" "${QFM_BUILD_ARGS[@]}"
fi
# Inspect the original container, not the isolated QFM overlay.
if ((!QFM_INSPECT)) && [[ -f "$QFM_DEPS/env.sh" ]]; then
    set +u
    source "$QFM_DEPS/env.sh"
    set -u
fi
exec python3 "$QFM_DIR/run.py" --output-dir /mnt/qfm-stage1-results "${QFM_RUN_ARGS[@]}"
