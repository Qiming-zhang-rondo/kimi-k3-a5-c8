#!/usr/bin/env bash
# Apply-only experimental K3 C8 DCP/DSpark installer for VA 0.29.
# No tests, dependency installation, model launch or service restart.
set -euo pipefail

release_commit='67a21fa3ada065039a749aa28cc52cc24c208f7b'
manifest_sha256='bff0d50e00a00a6cbb228540b92b201142f0e7648eefe748b7e2342237f51881'
release_url="https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/$release_commit"
manifest_name='k3-c8-dcp-dspark-manifest-20261029.json'
complete_name='k3-c8-dcp-dspark-complete-20261029.patch'
upgrade_name='k3-c8-dcp-dspark-upgrade-20261029.patch'
curl_options=(--fail --location --retry 2 --connect-timeout 15 --max-time 120)
source_dir=''
bundle_dir=''

usage() {
    echo 'Usage: bash install-c8-dcp-dspark.sh [--insecure] [--proxy URL] [--source VA_DIR] [--bundle-dir DIR]'
}

while (($#)); do
    case "$1" in
        --insecure) curl_options+=(--insecure); shift ;;
        --proxy) curl_options+=(--proxy "${2:?Missing proxy URL}"); shift 2 ;;
        --source) source_dir="${2:?Missing VA source directory}"; shift 2 ;;
        --bundle-dir) bundle_dir="${2:?Missing bundle directory}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

# Prefer the active model environment when both Python commands exist.
if command -v python >/dev/null 2>&1; then
    python_bin=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
else
    echo 'Python 3 is required; nothing changed.' >&2
    exit 1
fi

if [[ -z "$source_dir" ]]; then
    source_dir=$("$python_bin" - <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path

version = importlib.metadata.version("vllm-ascend")
if not version.startswith("0.29."):
    raise SystemExit(f"Expected installed vllm-ascend 0.29, got {version}; nothing changed.")
spec = importlib.util.find_spec("vllm_ascend")
if spec is None or not spec.origin:
    raise SystemExit("Cannot locate vllm-ascend source; pass --source VA_DIR.")
print(Path(spec.origin).resolve().parent.parent)
PY
    )
fi

if [[ -z "$bundle_dir" ]]; then
    bundle_dir=$(mktemp -d /tmp/k3-c8-dcp-dspark.XXXXXX)
    # Retain downloaded inputs for diagnosis; never remove a user-supplied directory.
    echo "Downloading pinned experimental bundle to $bundle_dir"
    for artifact in "$manifest_name" "$complete_name" "$upgrade_name"; do
        curl "${curl_options[@]}" "$release_url/$artifact" --output "$bundle_dir/$artifact"
    done
fi

"$python_bin" - "$source_dir" "$bundle_dir" "$manifest_sha256" "$manifest_name" "$complete_name" "$upgrade_name" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


class InstallError(Exception):
    pass


def fail(message):
    raise InstallError(message)


def read_regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        fail(f"Expected a regular file, refusing symlink or special file: {path}")
    return path.read_bytes()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def main():
    source_arg, bundle_arg, expected_manifest_sha, manifest_name, complete_name, upgrade_name = sys.argv[1:]
    bundle = Path(bundle_arg).resolve(strict=True)
    manifest_bytes = read_regular(bundle / manifest_name)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha):
        fail("Installer has no valid pinned manifest SHA256.")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
        fail("Manifest SHA256 mismatch.")
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    if not isinstance(manifest, dict):
        fail("Invalid manifest object.")
    expected_base = "b3d3bb6805ffaf8fcc20086d5e59dbf2352083b4"
    if manifest.get("base_commit") != expected_base:
        fail("Manifest does not describe the supported VA baseline.")
    for key in ("old_tree", "new_tree"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            fail(f"Invalid manifest {key}.")
    states = manifest.get("states")
    state_names = ("complete_before", "upgrade_before", "after")
    if not isinstance(states, dict) or any(not isinstance(states.get(name), dict) for name in state_names):
        fail("Manifest is missing source-state maps.")
    paths = sorted(states["after"])
    if not paths or any(set(states[name]) != set(paths) for name in state_names):
        fail("Manifest states must contain the same nonempty path set.")
    for relative in paths:
        if (
            not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in ("", ".", "..", ".git") for part in relative.split("/"))
            or PurePosixPath(relative).as_posix() != relative
        ):
            fail(f"Unsafe manifest path: {relative!r}")
        for name in state_names:
            entry = states[name][relative]
            if entry is not None and (
                not isinstance(entry, dict)
                or set(entry) != {"oid", "mode"}
                or entry.get("mode") not in ("100644", "100755")
                or not isinstance(entry.get("oid"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", entry["oid"])
            ):
                fail(f"Invalid file record in {name}: {relative}")
    patches = manifest.get("patches")
    if not isinstance(patches, dict):
        fail("Manifest is missing patch records.")
    for kind, filename in (("complete", complete_name), ("upgrade", upgrade_name)):
        entry = patches.get(kind)
        if (
            not isinstance(entry, dict)
            or entry.get("file") != filename
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            fail(f"Invalid {kind} patch record.")

    source = Path(source_arg).resolve(strict=True)

    def git(*args, input_bytes=None):
        result = subprocess.run(
            ["git", "--literal-pathspecs", "-C", str(source), *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.decode(errors="replace").strip()
            fail(f"Git {' '.join(args[:2])} failed: {detail}")
        return result.stdout

    root = Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip()).resolve(strict=True)
    if source != root:
        fail("VA source must be the Git repository root.")
    for relative in ("vllm_ascend/attention/mla_v1.py", "vllm_ascend/worker/v2/attn_utils.py"):
        if not (source / relative).is_file():
            fail("Not a compatible VA source tree.")

    def reject_staged_changes():
        staged = git("diff", "--cached", "--name-only", "-z", "--", *paths)
        if staged:
            names = ", ".join(os.fsdecode(item) for item in staged.split(b"\0") if item)
            fail(f"Affected paths contain staged changes; preserve or commit them before migration: {names}")

    def snapshot():
        actual = {}
        for relative in paths:
            target = source
            components = PurePosixPath(relative).parts
            missing_parent = False
            for part in components[:-1]:
                target = target / part
                try:
                    info = target.lstat()
                except FileNotFoundError:
                    missing_parent = True
                    break
                if not stat.S_ISDIR(info.st_mode):
                    fail(f"Affected path has a symlink or non-directory parent: {relative}")
            if missing_parent:
                actual[relative] = None
                continue
            target = target / components[-1]
            try:
                info = target.lstat()
            except FileNotFoundError:
                actual[relative] = None
                continue
            if not stat.S_ISREG(info.st_mode):
                fail(f"Affected path is a symlink or special file: {relative}")
            content = target.read_bytes()
            oid = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
            actual[relative] = {"oid": oid, "mode": "100755" if info.st_mode & stat.S_IXUSR else "100644"}
        return actual

    reject_staged_changes()
    before = snapshot()
    if before == states["after"]:
        print("Experimental K3 C8 DCP/DSpark bundle already installed; exact file state verified.")
        print(f"VA source: {source}")
        return
    matches = [kind for kind in ("complete", "upgrade") if before == states[f"{kind}_before"]]
    if len(matches) != 1:
        differences = {
            name: [path for path in paths if before[path] != states[name][path]]
            for name in ("complete_before", "upgrade_before")
        }
        for name, changed in differences.items():
            print(f"Does not match {name}: {', '.join(changed)}", file=sys.stderr)
        fail("Source contains unknown or partial changes in affected paths; refusing migration.")
    kind = matches[0]
    patch_entry = patches[kind]
    patch_bytes = read_regular(bundle / patch_entry["file"])
    if hashlib.sha256(patch_bytes).hexdigest() != patch_entry["sha256"]:
        fail(f"{kind.capitalize()} patch SHA256 mismatch.")

    # Both operations consume the same verified bytes, including offline bundles.
    git("apply", "--check", "-", input_bytes=patch_bytes)
    reject_staged_changes()
    if snapshot() != before:
        fail("Affected source files changed during preflight; refusing migration.")
    try:
        git("apply", "-", input_bytes=patch_bytes)
        final_matches = snapshot() == states["after"]
    except (InstallError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Patch application or final verification failed: {exc}. "
            "Source may have changed; do not force/reset. Preserve the tree for diagnosis."
        ) from exc
    if not final_matches:
        raise RuntimeError(
            "Patch was applied, but final file-state verification failed. "
            "Do not force/reset or rerun blindly; preserve the tree for diagnosis."
        )
    print(f"Installed experimental K3 C8 DCP/DSpark bundle via {kind} patch; final file state verified.")
    print(f"VA source: {source}")


try:
    main()
except (InstallError, OSError, ValueError) as exc:
    print(f"Installation refused: {exc}\nNo patch was applied; do not force/reset.", file=sys.stderr)
    sys.exit(1)
except RuntimeError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
PY

echo 'Experimental build: native A5, real DCP and DSpark model acceptance remain unverified.'
echo 'Use enable_sparse_sfa_c8=true and kv-cache-dtype auto. Restart your model service manually.'
echo 'Apply only: no tests, dependency installation, model launch or automatic restart.'
