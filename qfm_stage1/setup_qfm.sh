#!/usr/bin/env bash
# Private QFM staging only: no global pip install, CANN upgrade, or NPU run.
set -euo pipefail
pin=8b6924e68ed27181141cde84fa5a942c0c060aac
repo=https://gitcode.com/cann/ops-transformer.git
prefix=/mnt/k3-qfm-stage1-deps
cann_env= version_file= proxy= insecure=0 jobs=8
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
die() { echo "ERROR: $*" >&2; exit 2; }
usage() {
    echo "Usage: bash setup_qfm.sh [--prefix DIR] [--cann-env FILE] [--cann-version-file FILE]"
    echo "       [--jobs N] [--proxy URL] [--insecure]"
    echo "Default prefix: /mnt/k3-qfm-stage1-deps; existing CANN must be exactly 9.2.0."
    echo "Autodetects existing CANN paths in priority order and rejects conflicting toolkit versions."
    echo "Builds pinned QFM+metadata, stages private OPP/wheels, writes env.sh + build-manifest.json."
    echo "Refuses occupied staging directories or dirty/unknown source; never resets or upgrades anything."
    echo "--proxy covers clone/build downloads; --insecure disables git TLS verification in those child processes only."
    echo "CMake URL-download TLS is not overridden; failures remain errors. No global trust/proxy configuration changes."
    echo "Activate env.sh only inside a dedicated child shell; no hardware validation is performed."
}
while (($#)); do
    case "$1" in
        --prefix) prefix=${2:?missing prefix}; shift 2 ;;
        --cann-env) cann_env=${2:?missing CANN environment}; shift 2 ;;
        --cann-version-file) version_file=${2:?missing version record}; shift 2 ;;
        --jobs) jobs=${2:?missing jobs}; shift 2 ;;
        --proxy) proxy=${2:?missing proxy}; shift 2 ;;
        --insecure) insecure=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown argument: $1" ;;
    esac
done
[[ "$prefix" == /* && "$prefix" != / && "$jobs" =~ ^[1-9][0-9]*$ ]] || die "Need an absolute dedicated prefix and positive jobs."
command -v realpath >/dev/null || die "realpath is required."
[[ -f "$script_dir/build_qfm.sh" ]] || die "Adjacent build_qfm.sh is missing."
# Prefer the active root, then its wrapper; explicit paths override discovery.
choose_file() {
    local candidate
    for candidate in "$@"; do
        if [[ -f "$candidate" ]]; then realpath "$candidate"; return; fi
    done
    die "Missing $selection; supply its explicit path."
}
if [[ -z "$cann_env" ]]; then
    candidates=()
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        for path in "$ASCEND_HOME_PATH/set_env.sh" "$ASCEND_HOME_PATH/../set_env.sh"; do
            [[ ! -f "$path" ]] || candidates+=("$path")
        done
    fi
    if ((${#candidates[@]} == 0)); then
        candidates=(/usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh /usr/local/Ascend/latest/set_env.sh)
    fi
    selection="CANN environment (--cann-env)"; cann_env=$(choose_file "${candidates[@]}")
else
    [[ -f "$cann_env" ]] || die "CANN environment does not exist."
    cann_env=$(realpath "$cann_env")
fi
set +u
source "$cann_env"
set -u
[[ -n "${ASCEND_HOME_PATH:-}" && -d "$ASCEND_HOME_PATH" ]] || die "CANN environment did not select an existing ASCEND_HOME_PATH."
cann_home=$(realpath "$ASCEND_HOME_PATH"); cann_parent=$(dirname "$cann_home")
version_candidates=("$cann_home/version.cfg" "$cann_home/version.info" "$cann_home/aarch64-linux/ascend_toolkit_install.info" "$cann_home/x86_64-linux/ascend_toolkit_install.info" "$cann_parent/ascend_toolkit_install.info")
if [[ -z "$version_file" ]]; then
    selection="toolkit version record (--cann-version-file)"
    version_file=$(choose_file "${version_candidates[@]}")
else
    [[ -f "$version_file" ]] || die "Version record does not exist."
    version_file=$(realpath "$version_file")
fi
[[ "$version_file" == "$cann_home/"* || "$version_file" == "$cann_parent/ascend_toolkit_install.info" ]] || die "Version evidence is outside the selected CANN toolkit."
found_version=0
for record in "$version_file" "${version_candidates[@]}"; do
    [[ -f "$record" ]] || continue
    while IFS= read -r version; do
        [[ "$version" =~ ^9\.2\.0(\.[0-9]+)*$ ]] || die "Conflicting/non-9.2.0 CANN version in $record: $version"
        found_version=1
    done < <(awk -F= 'tolower($1) ~ /^[[:space:]]*(version|cann_version|toolkit_version|version_name)[[:space:]]*$/ {gsub(/[[:space:]\047\042]/,"",$2); print $2}' "$record")
done
((found_version)) || die "No CANN 9.2.0 version evidence found."
[[ ! -L "$prefix" ]] || die "Prefix must not be a symlink."
mkdir -p "$prefix"; prefix=$(cd "$prefix" && pwd -P)
shopt -s nullglob dotglob
for path in "$prefix"/*; do [[ "$path" == "$prefix/sources" && -d "$path" && ! -L "$path" ]] || die "Occupied or unknown prefix entry: $path"; done
mkdir -p "$prefix/sources"
source_dir="$prefix/sources/ops-transformer"
metadata_source="$prefix/sources/ops-transformer-metadata"
for path in "$prefix/sources"/*; do [[ ( "$path" == "$source_dir" || "$path" == "$metadata_source" ) && ! -L "$path" ]] || die "Unknown source entry: $path"; done
git_download() {
    local options=(-c http.sslVerify=true)
    [[ -z "$proxy" ]] || options+=(-c "http.proxy=$proxy")
    ((insecure == 0)) || options+=(-c http.sslVerify=false)
    env -u GIT_SSL_NO_VERIFY git "${options[@]}" "$@"
}
((insecure == 0)) || echo "WARNING: Explicit git TLS verification bypass applies to clone/build child downloads; not global trust or CMake URL downloads." >&2
for checkout in "$source_dir" "$metadata_source"; do
    if [[ ! -e "$checkout" ]]; then
        git_download clone --filter=blob:none --no-checkout "$repo" "$checkout"
        git_download -C "$checkout" fetch --depth=1 origin "$pin"
        git_download -C "$checkout" checkout --detach "$pin"
    fi
    [[ -d "$checkout/.git" && -f "$checkout/build.sh" ]] || die "Source is not a dedicated ops-transformer checkout."
    [[ $(git -C "$checkout" rev-parse HEAD) == "$pin" ]] || die "Existing source is not the exact pinned commit; refusing to reset."
    source_status=$(git -C "$checkout" status --porcelain --untracked-files=all --ignored=matching)
    [[ -z "$source_status" ]] || die "Source is dirty or contains unknown/ignored files; refusing to clean it."
done
build_env=(env -u GIT_SSL_NO_VERIFY)
[[ -z "$proxy" ]] || build_env+=("HTTP_PROXY=$proxy" "HTTPS_PROXY=$proxy" "http_proxy=$proxy" "https_proxy=$proxy")
((insecure == 0)) || build_env+=(GIT_SSL_NO_VERIFY=true)
"${build_env[@]}" bash "$script_dir/build_qfm.sh" --source "$source_dir" --metadata-source "$metadata_source" --cann-env "$cann_env" --cann-version-file "$version_file" --jobs "$jobs"
qfm_runs=("$source_dir"/build_out/cann-ops-transformer-qfm_stage1_linux-*.run)
meta_runs=("$metadata_source"/build_out/cann-ops-transformer-qfm_meta_stage1_linux-*.run)
base_wheels=("$source_dir"/build_out/cann_ops_transformer-*.whl)
qfm_wheels=("$source_dir"/build_out/cann_ops_transformer_qfm_stage1-*.whl)
((${#qfm_runs[@]} == 1 && ${#meta_runs[@]} == 1 && ${#base_wheels[@]} == 1 && ${#qfm_wheels[@]} == 1)) || die "Expected two distinct-vendor native packages and one of each frontend wheel."
mkdir "$prefix/opp" "$prefix/python"
for package in "${qfm_runs[0]}" "${meta_runs[0]}"; do sh "$package" "--install-path=$prefix/opp" 2>&1 | tee -a "$prefix/install.log"; done
vendor_envs=()
for vendor in qfm_stage1 qfm_meta_stage1; do
    path="$prefix/opp/vendors/${vendor}_transformer/bin/set_env.bash"
    [[ -f "$path" ]] || die "Missing private vendor environment: $path"
    path=$(realpath "$path"); [[ "$path" == "$prefix/opp/"* ]] || die "Vendor environment escapes private prefix."
    vendor_envs+=("$path")
done
python3 - "$prefix/opp/vendors" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); qfm = root / "qfm_stage1_transformer"; meta = root / "qfm_meta_stage1_transformer"
kernel = qfm / "op_impl/ai_core/tbe/kernel"
if not (kernel / "config/ascend950/quant_flash_mla.json").is_file(): raise SystemExit("Missing QFM kernel config")
if not any(p.is_file() and p.stat().st_size and p.suffix in (".o", ".bin") for p in (kernel / "ascend950/quant_flash_mla").glob("**/*")): raise SystemExit("Missing QFM native kernel binary")
for path in [qfm / "op_api/lib/libcust_opapi.so", meta / "op_api/lib/libcust_opapi.so", meta / "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so"]:
    if not path.is_file() or path.stat().st_size == 0: raise SystemExit(f"Missing native library: {path}")
config = json.loads((meta / "op_impl/cpu/config/cust_aicpu_kernel.json").read_text())
if config["QuantFlashMlaMetadata"]["opInfo"]["kernelSo"] != "libtransformer_aicpu_kernels.so": raise SystemExit("Metadata library registration mismatch")
PY
python3 -m pip install --no-deps --no-index --target "$prefix/python" "${base_wheels[0]}" "${qfm_wheels[0]}" 2>&1 | tee "$prefix/pip.log"
# printf %q preserves spaces/metacharacters without evaluating path contents.
{
    echo '# Source only inside a dedicated child shell, never a running model environment.'
    echo '[[ ${BASH_SOURCE[0]} != "$0" ]] || { echo "Source this file inside a child shell." >&2; exit 2; }'
    echo '_qfm_stage1_nounset=0; [[ $- != *u* ]] || _qfm_stage1_nounset=1; set +u'
    printf 'if ! source %q || ! source %q || ! source %q; then\n' "$cann_env" "${vendor_envs[0]}" "${vendor_envs[1]}"
    echo '  ((_qfm_stage1_nounset == 0)) || set -u; unset _qfm_stage1_nounset; return 1; fi'
    printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$prefix/python"
    echo '((_qfm_stage1_nounset == 0)) || set -u; unset _qfm_stage1_nounset'
} > "$prefix/env.sh"
python3 - "$prefix" "$pin" "$repo" "$source_dir" "$cann_env" "$version_file" "$insecure" "${qfm_runs[0]}" "${meta_runs[0]}" "${base_wheels[0]}" "${qfm_wheels[0]}" <<'PY'
import datetime, hashlib, json, pathlib, sys
prefix, pin, repo, source, cann_env, version, insecure, *artifacts = sys.argv[1:]
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
manifest = {"schema": 1, "status": "staged_not_hardware_validated", "source_commit": pin,
            "source_url": repo, "source_dir": source, "cann_env": cann_env,
            "metadata_source_dir": str(pathlib.Path(prefix) / "sources/ops-transformer-metadata"),
            "version_file": version, "version_sha256": digest(version),
            "vendor_envs": [str(pathlib.Path(prefix) / "opp/vendors" / (v + "_transformer/bin/set_env.bash")) for v in ("qfm_stage1", "qfm_meta_stage1")],
            "python_target": str(pathlib.Path(prefix) / "python"), "insecure_git_download": insecure == "1",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "artifacts": [{"path": path, "sha256": digest(path)} for path in artifacts]}
pathlib.Path(prefix, "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "STAGED (not hardware validated): $prefix"
echo "Activate $prefix/env.sh only in the probe child shell. Existing staging is never overwritten."
