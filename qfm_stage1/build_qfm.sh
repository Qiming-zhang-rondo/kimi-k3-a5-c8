#!/usr/bin/env bash
# Compile only QFM + its AICPU metadata kernel. No installation or CANN upgrade.
set -euo pipefail

pin=8b6924e68ed27181141cde84fa5a942c0c060aac
source_dir=
metadata_dir=
cann_env=
version_file=
jobs=8

usage() {
    echo "Usage: bash build_qfm.sh --source BASE --metadata-source META --cann-env PATH --cann-version-file PATH [--jobs N]"
    echo "Use two dedicated checkouts/worktrees at $pin and existing CANN 9.2.0."
    echo "Outputs stay in BASE/build_out and META/build_out; nothing is installed. See BUILD.md."
}

while (($#)); do
    case "$1" in
        --source) source_dir=${2:?missing source path}; shift 2 ;;
        --metadata-source) metadata_dir=${2:?missing metadata source path}; shift 2 ;;
        --cann-env) cann_env=${2:?missing CANN environment script}; shift 2 ;;
        --cann-version-file) version_file=${2:?missing toolkit version record}; shift 2 ;;
        --jobs) jobs=${2:?missing job count}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

for tree in "$source_dir" "$metadata_dir"; do
    [[ -e "$tree/.git" && -f "$tree/build.sh" ]] || {
        echo "Two dedicated checked-out ops-transformer trees are required (worktrees are supported)." >&2; exit 2;
    }
done
[[ -f "$cann_env" && -f "$version_file" && "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    usage >&2; exit 2;
}
source_dir=$(cd "$source_dir" && pwd -P)
metadata_dir=$(cd "$metadata_dir" && pwd -P)
[[ "$source_dir" != "$metadata_dir" ]] || { echo "QFM and metadata need distinct build trees." >&2; exit 2; }
for tree in "$source_dir" "$metadata_dir"; do
    [[ $(git -C "$tree" rev-parse HEAD) == "$pin" ]] || {
        echo "Both source trees must be pinned to $pin." >&2; exit 2;
    }
    git -C "$tree" diff --quiet HEAD -- || {
        echo "Tracked source modifications found in $tree; use unmodified pinned trees." >&2; exit 2;
    }
done
# Vendor environment scripts are not necessarily nounset-safe.
set +u
source "$cann_env"
set -u
[[ -n "${ASCEND_HOME_PATH:-}" ]] || { echo "CANN environment did not set ASCEND_HOME_PATH." >&2; exit 2; }
python3 - "$version_file" "$ASCEND_HOME_PATH" <<'PY'
import pathlib
import re
import sys
record, active = (pathlib.Path(p).resolve() for p in sys.argv[1:])
parent_records = {"version.cfg", "version.info", "ascend_toolkit_install.info", "toolkit_install.info"}
bound = record.is_relative_to(active) or (record.parent == active.parent and record.name in parent_records)
if not bound:
    raise SystemExit("Version evidence must belong to active ASCEND_HOME_PATH or be its direct-parent toolkit record.")
text = record.read_text(errors="replace")
versions = re.findall(r"(?im)^\s*(?:version|cann_version|cannversion|version_name)\s*[:=]\s*[\"']?(\d+\.\d+\.\d+)", text)
if not versions:
    versions = re.findall(r"(?m)^\s*(\d+\.\d+\.\d+)(?:\s|$)", text)
if not versions or set(versions) != {"9.2.0"}:
    raise SystemExit(f"Active toolkit version must be exactly 9.2.0; parsed {versions!r} from {record}.")
print(f"Verified active toolkit record: {record} -> 9.2.0")
PY
for command in cmake make gcc g++ patch; do
    command -v "$command" >/dev/null || { echo "Missing build prerequisite: $command (not auto-installed)." >&2; exit 2; }
done
python3 -c 'import build, torch, torch_npu' || {
    echo "Build prerequisites missing. Use the existing matching Python environment; no packages were installed." >&2
    exit 2
}

cd "$source_dir"
log_dir="$source_dir/stage1-build-logs"
mkdir -p "$log_dir"
echo "Pinned source: $pin"
echo "CANN environment: $cann_env"
echo "CANN toolkit: $ASCEND_HOME_PATH"
echo "Version evidence: $version_file"
echo "Build-only: no .run installer, pip install, or CANN upgrade will be executed."

# In this pin ENABLE_AICPU_KERNEL suppresses normal AICore generation globally.
# Separate trees and vendors prevent both missing QFM binaries and package replacement.
bash build.sh --pkg --soc=ascend950 --vendor_name=qfm_stage1 \
    --ops=quant_flash_mla "-j$jobs" \
    2>&1 | tee "$log_dir/qfm-package.log"
(
    cd "$metadata_dir"
    bash build.sh --pkg --soc=ascend950 --vendor_name=qfm_meta_stage1 \
        --ops=quant_flash_mla_metadata --opkernel_aicpu "-j$jobs"
) 2>&1 | tee "$log_dir/metadata-package.log"
if grep -Eiq 'all requested ops skipped|No operator can be compiled|skipped by CANN version check' \
        "$log_dir/qfm-package.log" "$log_dir/metadata-package.log"; then
    echo "Build skipped requested kernels on this CANN version; not a successful QFM build." >&2
    exit 1
fi

# Pure-Python frontend wheel, not additional native operators. Needed for an
# isolated PYTHONPATH prefix even when the running model has an older frontend.
bash build.sh --torch_extension 2>&1 | tee "$log_dir/python-base.log"
# quant_flash_mla's frontend registration also defines the metadata wrapper.
bash build.sh --torch_extension --ops=quant_flash_mla \
    --vendor_name=qfm_stage1 2>&1 | tee "$log_dir/python-qfm.log"
shopt -s nullglob
run_files=(build_out/cann-ops-transformer-qfm_stage1_linux-*.run)
metadata_run_files=("$metadata_dir"/build_out/cann-ops-transformer-qfm_meta_stage1_linux-*.run)
base_wheels=(build_out/cann_ops_transformer-*.whl)
qfm_wheels=(build_out/cann_ops_transformer_qfm_stage1-*.whl)
if ((${#run_files[@]} != 1 || ${#metadata_run_files[@]} != 1 || ${#base_wheels[@]} != 1 || ${#qfm_wheels[@]} != 1)); then
    echo "Expected one .run per vendor and base/QFM frontend wheels; inspect both build_out directories." >&2
    exit 1
fi
echo "BUILD COMPLETE; runtime/NPU correctness is NOT yet validated. No packages installed."
printf '%s\n' "${run_files[@]}" "${metadata_run_files[@]}" "${base_wheels[@]}" "${qfm_wheels[@]}"
