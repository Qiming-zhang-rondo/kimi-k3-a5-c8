"""CPU-only filesystem/command-flow tests; no downloads, pip install, or NPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PIN = "8b6924e68ed27181141cde84fa5a942c0c060aac"
SCRIPT = Path(__file__).with_name("setup_qfm.sh")


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(SCRIPT, tools / SCRIPT.name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    trace = tmp_path / "trace.jsonl"
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}", TRACE=str(trace))
    cann = tmp_path / "cann"
    toolkit = cann / "latest"
    toolkit.mkdir(parents=True)
    cann_env = cann / "set_env.sh"
    cann_env.write_text(f"export ASCEND_HOME_PATH='{toolkit}'\n")
    version = toolkit / "version.cfg"
    version.write_text("version=9.2.0\n")
    env["ASCEND_HOME_PATH"] = str(toolkit)
    prefix = tmp_path / "private stage"
    executable(binaries / "git", f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['TRACE'], 'a') as f:
    f.write(json.dumps({'tool':'git','args':args,'ssl_env':os.environ.get('GIT_SSL_NO_VERIFY')}) + '\\n')
while args[:1] == ['-c']:
    args = args[2:]
if args[:1] == ['-C']:
    args = args[2:]
if args[0] == 'clone':
    path = pathlib.Path(args[-1]); (path / '.git').mkdir(parents=True)
    (path / 'build.sh').write_text('# fake pinned source\\n')
elif args[0] == 'rev-parse':
    print(os.environ.get('FAKE_PIN', '8b6924e68ed27181141cde84fa5a942c0c060aac'))
elif args[0] == 'status':
    print(os.environ.get('FAKE_DIRTY', ''), end='')
elif args[0] not in ('fetch','checkout'):
    raise SystemExit('Unexpected git command: ' + repr(args))
''')
    executable(binaries / "python3", f"#!{sys.executable}\n" + f'''
import json, os, pathlib, sys
args = sys.argv[1:]
if args[:2] == ['-m', 'pip']:
    with open(os.environ['TRACE'], 'a') as f:
        f.write(json.dumps({{'tool':'pip','args':args}}) + '\\n')
    target = pathlib.Path(args[args.index('--target') + 1])
    (target / 'private_frontend.py').write_text('# mock wheel installed\\n')
else:
    os.execv({sys.executable!r}, [{sys.executable!r}] + args)
''')
    # The stand-in build wrapper generates stand-in installers, not real code.
    executable(tools / "build_qfm.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + '''
source_dir= metadata_source=
while (($#)); do
    case "$1" in --source) source_dir=$2; shift 2;; --metadata-source) metadata_source=$2; shift 2;; *) shift;; esac
done
python3 -c 'import json,os; open(os.environ["TRACE"],"a").write(json.dumps({"tool":"build","args":[],"proxy":os.getenv("HTTPS_PROXY"),"ssl_env":os.getenv("GIT_SSL_NO_VERIFY")}) + "\\n")'
[[ ${FAKE_BUILD_FAIL:-0} == 0 ]] || exit 17
mkdir -p "$source_dir/build_out" "$metadata_source/build_out"
for vendor in qfm_stage1 qfm_meta_stage1; do
output=$source_dir; [[ "$vendor" != qfm_meta_stage1 ]] || output=$metadata_source
cat > "$output/build_out/cann-ops-transformer-${vendor}_linux-fake.run" <<'INSTALL'
#!/bin/sh
set -eu
target=${1#--install-path=}
vendor=$(basename "$0"); vendor=${vendor#cann-ops-transformer-}; vendor=${vendor%_linux-fake.run}
mkdir -p "$target/vendors/${vendor}_transformer/bin"
if [ "${FAKE_NO_VENDOR_ENV:-0}" != 1 ]; then
    printf 'export MOCK_VENDOR_READY=1\\n' > "$target/vendors/${vendor}_transformer/bin/set_env.bash"
fi
mkdir -p "$target/vendors/${vendor}_transformer/op_impl/ai_core/tbe/kernel/ascend950/quant_flash_mla"
if [ "${FAKE_NO_KERNEL:-0}" != 1 ]; then
    printf 'mock ELF' > "$target/vendors/${vendor}_transformer/op_impl/ai_core/tbe/kernel/ascend950/quant_flash_mla/QuantFlashMla_fake.o"
fi
mkdir -p "$target/vendors/${vendor}_transformer/op_impl/ai_core/tbe/kernel/config/ascend950"
printf '{}' > "$target/vendors/${vendor}_transformer/op_impl/ai_core/tbe/kernel/config/ascend950/quant_flash_mla.json"
mkdir -p "$target/vendors/${vendor}_transformer/op_impl/cpu/aicpu_kernel/impl"
printf 'mock ELF' > "$target/vendors/${vendor}_transformer/op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so"
mkdir -p "$target/vendors/${vendor}_transformer/op_api/lib" "$target/vendors/${vendor}_transformer/op_impl/cpu/config"
printf 'mock ELF' > "$target/vendors/${vendor}_transformer/op_api/lib/libcust_opapi.so"
printf '{"QuantFlashMlaMetadata":{"opInfo":{"kernelSo":"libtransformer_aicpu_kernels.so"}}}' > "$target/vendors/${vendor}_transformer/op_impl/cpu/config/cust_aicpu_kernel.json"
INSTALL
done
printf 'base' > "$source_dir/build_out/cann_ops_transformer-1-py3-none-any.whl"
printf 'qfm' > "$source_dir/build_out/cann_ops_transformer_qfm_stage1-1-py3-none-any.whl"
if [[ ${FAKE_EXTRA_WHEEL:-0} == 1 ]]; then
    printf 'extra' > "$source_dir/build_out/cann_ops_transformer-2-py3-none-any.whl"
fi
''')
    return {"root": tmp_path, "tools": tools, "env": env, "prefix": prefix,
            "cann": cann, "toolkit": toolkit, "cann_env": cann_env,
            "version": version, "trace": trace}


def invoke(sandbox, *args):
    return subprocess.run(
        ["bash", str(sandbox["tools"] / "setup_qfm.sh"), "--prefix", str(sandbox["prefix"]), *args],
        env=sandbox["env"], text=True, capture_output=True, timeout=8,
    )


def events(sandbox):
    return [json.loads(line) for line in sandbox["trace"].read_text().splitlines()] if sandbox["trace"].exists() else []


def preexisting_source(sandbox):
    for name in ("ops-transformer", "ops-transformer-metadata"):
        source = sandbox["prefix"] / "sources" / name
        (source / ".git").mkdir(parents=True)
        (source / "build.sh").write_text("# existing pinned checkout\n")


def test_help_and_shell_syntax():
    result = subprocess.run(["bash", str(SCRIPT), "--help"], text=True, capture_output=True, timeout=3)
    assert result.returncode == 0
    assert "--prefix" in result.stdout and "--insecure" in result.stdout
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    assert len(SCRIPT.read_text().splitlines()) <= 200


def test_private_staging_and_child_activation(sandbox):
    result = invoke(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = sandbox["prefix"]
    manifest = json.loads((prefix / "build-manifest.json").read_text())
    assert manifest["source_commit"] == PIN
    assert manifest["status"] == "staged_not_hardware_validated"
    assert manifest["python_target"] == str(prefix / "python")
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    pip = [event for event in events(sandbox) if event["tool"] == "pip"]
    assert len(pip) == 1
    assert "--no-deps" in pip[0]["args"] and "--no-index" in pip[0]["args"]
    assert pip[0]["args"][pip[0]["args"].index("--target") + 1] == str(prefix / "python")
    assert "--upgrade" not in pip[0]["args"]
    child = subprocess.run(
        ["bash", "-uc", 'source "$1/env.sh"; printf "%s\\n%s\\n" "$PYTHONPATH" "$MOCK_VENDOR_READY"', "bash", str(prefix)],
        env=dict(sandbox["env"], PYTHONPATH="/existing/python"), text=True, capture_output=True, timeout=3,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == [str(prefix / "python") + ":/existing/python", "1"]
    assert not (sandbox["toolkit"] / "opp").exists()
    assert invoke(sandbox).returncode != 0  # never overwrite a previous stage


def test_clean_exact_pin_reuses_source_without_download(sandbox):
    preexisting_source(sandbox)
    result = invoke(sandbox)
    assert result.returncode == 0, result.stderr
    assert not any("clone" in item["args"] or "fetch" in item["args"] for item in events(sandbox))


@pytest.mark.parametrize("key,value", [("FAKE_PIN", "unknown"), ("FAKE_DIRTY", " M build.sh\n"), ("FAKE_DIRTY", "!! build_out\n")])
def test_dirty_unknown_source_is_never_reset(sandbox, key, value):
    preexisting_source(sandbox)
    sandbox["env"][key] = value
    result = invoke(sandbox)
    assert result.returncode != 0
    assert not (sandbox["prefix"] / "opp").exists()
    assert not any("reset" in item["args"] or "checkout" in item["args"] for item in events(sandbox))


@pytest.mark.parametrize("record", ["version=9.2.1", "version=9.3.0", "version=9.2.0RC1", "version=9.2.0\ncann_version=9.3.0", "unknown"])
def test_wrong_or_ambiguous_version_rejected_before_download(sandbox, record):
    sandbox["version"].write_text(record)
    result = invoke(sandbox)
    assert result.returncode != 0
    assert events(sandbox) == []
    assert not sandbox["prefix"].exists()


def test_version_evidence_from_other_toolkit_is_rejected(sandbox):
    other = sandbox["root"] / "other-version.cfg"
    other.write_text("version=9.2.0\n")
    result = invoke(sandbox, "--cann-version-file", str(other))
    assert result.returncode != 0
    assert "outside the selected" in result.stderr
    assert events(sandbox) == []


def test_matching_version_records_and_env_wrappers_use_priority(sandbox):
    second_env = sandbox["toolkit"] / "set_env.sh"
    second_env.write_text(sandbox["cann_env"].read_text())
    (sandbox["toolkit"] / "version.info").write_text("version=9.2.0\n")
    sandbox["version"].write_text("Version=9.2.0\nunrelated_compiler=2.3.4\n")
    result = invoke(sandbox)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((sandbox["prefix"] / "build-manifest.json").read_text())
    assert manifest["cann_env"] == str(second_env)
    assert manifest["version_file"] == str(sandbox["version"])


def test_conflicting_active_version_records_fail(sandbox):
    (sandbox["toolkit"] / "version.info").write_text("version=9.3.0\n")
    assert invoke(sandbox).returncode != 0
    assert events(sandbox) == []


@pytest.mark.parametrize("entry", ["python", "opp", "env.sh", "unrelated"])
def test_occupied_prefix_is_not_overwritten(sandbox, entry):
    sandbox["prefix"].mkdir()
    occupied = sandbox["prefix"] / entry
    occupied.write_text("preserve me")
    assert invoke(sandbox).returncode != 0
    assert occupied.read_text() == "preserve me"
    assert events(sandbox) == []


def test_proxy_and_insecure_are_scoped_to_git_download(sandbox):
    sandbox["env"]["GIT_SSL_NO_VERIFY"] = "1"
    result = invoke(sandbox, "--proxy", "http://localhost:4444", "--insecure")
    assert result.returncode == 0, result.stderr
    downloads = [event for event in events(sandbox) if "http.proxy=http://localhost:4444" in event["args"]]
    assert len(downloads) == 6
    assert all("http.sslVerify=false" in event["args"] and event["ssl_env"] is None for event in downloads)
    assert all("http.proxy=http://localhost:4444" not in event["args"] for event in events(sandbox) if event["tool"] == "pip")
    assert "localhost:4444" not in (sandbox["prefix"] / "build-manifest.json").read_text()
    build = next(event for event in events(sandbox) if event["tool"] == "build")
    assert build["proxy"] == "http://localhost:4444"
    assert build["ssl_env"] == "true"


@pytest.mark.parametrize("variable", ["FAKE_NO_VENDOR_ENV", "FAKE_EXTRA_WHEEL", "FAKE_BUILD_FAIL", "FAKE_NO_KERNEL"])
def test_incomplete_build_or_install_never_writes_success_manifest(sandbox, variable):
    sandbox["env"][variable] = "1"
    sandbox["env"]["PYTHONOPTIMIZE"] = "1"  # artifact checks must not be assert statements
    result = invoke(sandbox)
    assert result.returncode != 0
    assert not (sandbox["prefix"] / "build-manifest.json").exists()
    assert not any(event["tool"] == "pip" for event in events(sandbox))
