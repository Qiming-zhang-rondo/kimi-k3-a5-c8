"""CPU-only supervisor tests using disposable stand-in worker processes."""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import run as runner


def _worker(tmp_path, source, name="worker.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return [sys.executable, "-u", str(path)]


def _run(tmp_path, source, *, timeout=3.0):
    output = io.StringIO()
    result = runner.run_child(
        _worker(tmp_path, source),
        "small",
        tmp_path / "small.log",
        timeout_seconds=timeout,
        env=runner.child_environment(7),
        stdout=output,
    )
    return result, output.getvalue()


def _summaries(directory):
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("*/summary.json"))
    ]


def test_explicit_success_is_logged_and_streamed(tmp_path):
    result, output = _run(
        tmp_path, 'print(\'{"event":"suite_complete","suite":"small","pass":true}\')'
    )
    assert result["status"] == "passed"
    assert result["returncode"] == 0
    assert result["completion"]["suite"] == "small"
    assert Path(result["log_path"]).read_text() == output


@pytest.mark.parametrize(
    "source", ["pass", "print('some progress')", "print('{bad JSON}')"]
)
def test_silent_or_missing_completion_is_incomplete(tmp_path, source):
    result, _ = _run(tmp_path, source)
    assert result["returncode"] == 0
    assert result["status"] == "incomplete"
    assert result["reason"] == "missing_or_ambiguous_suite_complete"


@pytest.mark.parametrize(
    "event",
    [
        {"event": "suite_complete", "suite": "other", "pass": True},
        {"event": "suite_complete", "suite": "small", "pass": 1},
        {"event": "suite_complete", "suite": "small"},
    ],
)
def test_mismatched_or_nonboolean_completion_is_incomplete(tmp_path, event):
    result, _ = _run(tmp_path, f"print({json.dumps(json.dumps(event))})")
    assert result["status"] == "incomplete"


def test_duplicate_completion_is_incomplete(tmp_path):
    line = 'print(\'{"event":"suite_complete","suite":"small","pass":true}\')\n'
    result, _ = _run(tmp_path, line * 2)
    assert result["status"] == "incomplete"


def test_success_marker_does_not_override_nonzero_exit(tmp_path):
    result, _ = _run(
        tmp_path,
        'print(\'{"event":"suite_complete","suite":"small","pass":true}\')\nraise SystemExit(4)',
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 4
    assert result["reason"] == "nonzero_exit"


def test_failed_check_cannot_be_overridden_by_completion(tmp_path):
    source = (
        'print(\'{"event":"check","pass":false}\')\n'
        'print(\'{"event":"suite_complete","suite":"small","pass":true}\')'
    )
    result, _ = _run(tmp_path, source)
    assert result["status"] == "failed"
    assert result["reason"] == "worker_reported_failure"


def test_completion_without_final_newline(tmp_path):
    result, _ = _run(
        tmp_path,
        'import sys\nsys.stdout.write(\'{"event":"suite_complete","suite":"small","pass":true}\')',
    )
    assert result["status"] == "passed"


def test_large_unterminated_log_does_not_count_as_completion(tmp_path):
    result, output = _run(tmp_path, "import sys\nsys.stdout.write('x' * 1100000)")
    assert result["status"] == "incomplete"
    assert len(output) == 1100000


def test_timeout_stops_worker_and_never_accepts_early_success(tmp_path):
    started = time.monotonic()
    result, output = _run(
        tmp_path,
        'import time\nprint(\'{"event":"suite_complete","suite":"small","pass":true}\')\ntime.sleep(30)',
        timeout=0.15,
    )
    assert time.monotonic() - started < 4
    assert result["status"] == "incomplete"
    assert result["reason"] == "timeout"
    assert result["returncode"] is not None
    assert "device may still be hung" in output


@pytest.mark.skipif(
    os.name != "posix", reason="process-group contract targets Linux/macOS"
)
def test_timeout_kills_descendant_process_group(tmp_path):
    marker = tmp_path / "descendant_survived"
    child_source = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(marker)!r}).touch()"
    source = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "time.sleep(30)\n"
    )
    result, _ = _run(tmp_path, source, timeout=0.15)
    assert result["status"] == "incomplete"
    time.sleep(1.6)
    assert not marker.exists()


def test_missing_worker_is_failure(tmp_path):
    result = runner.run_child(
        [sys.executable, str(tmp_path / "missing.py")],
        "small",
        tmp_path / "missing.log",
        timeout_seconds=2,
        env=runner.child_environment(0),
        stdout=io.StringIO(),
    )
    assert result["status"] == "failed"
    assert result["returncode"] != 0


def test_child_device_mapping_removes_conflicting_id():
    original = {
        "ASCEND_DEVICE_ID": "9",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "KEEP": "yes",
    }
    env = runner.child_environment(7, original)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "7"
    assert "ASCEND_DEVICE_ID" not in env
    assert env["KEEP"] == "yes"
    assert original["ASCEND_DEVICE_ID"] == "9"


def test_worker_observes_only_selected_physical_card(tmp_path):
    command = _worker(
        tmp_path,
        (
            "import os, json\n"
            "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '7'\n"
            "assert 'ASCEND_DEVICE_ID' not in os.environ\n"
            "print(json.dumps({'event':'suite_complete','suite':'small','pass':True}))\n"
        ),
    )
    environment = runner.child_environment(7, {**os.environ, "ASCEND_DEVICE_ID": "99"})
    result = runner.run_child(
        command,
        "small",
        tmp_path / "device.log",
        timeout_seconds=2,
        env=environment,
        stdout=io.StringIO(),
    )
    assert result["status"] == "passed"


def test_all_job_order_and_performance_forwarding():
    args = runner.parse_args(
        [
            "--card",
            "7",
            "--queries",
            "32",
            "--kv-length",
            "129",
            "--heads",
            "24",
            "--iterations",
            "3",
            "--prolog-source",
            "/tmp/reference",
        ]
    )
    jobs = runner.build_jobs(args)
    assert [job["suite"] for job in jobs] == [
        "preflight",
        "small",
        "graph",
        "dcp",
        "performance",
    ]
    for job in jobs:
        index = job["command"].index("--card")
        assert job["command"][index + 1] == "0"
    performance = jobs[-1]["command"]
    for flag, value in (
        ("--queries", "32"),
        ("--kv-length", "129"),
        ("--heads", "24"),
        ("--iterations", "3"),
    ):
        assert performance[performance.index(flag) + 1] == value
    args.suite = "prolog"
    prolog_jobs = runner.build_jobs(args)
    assert "--preflight" in prolog_jobs[0]["command"]
    assert all("--prolog-source" in job["command"] for job in prolog_jobs)


@pytest.mark.parametrize(
    "suite,expected",
    [
        ("inspect", ["inspect"]),
        ("preflight", ["preflight"]),
        ("small", ["preflight", "small"]),
        ("graph", ["preflight", "graph"]),
        ("dcp", ["preflight", "dcp"]),
        ("prolog", ["prolog_preflight", "prolog"]),
        ("performance", ["preflight", "performance"]),
    ],
)
def test_individual_suite_prerequisites(suite, expected):
    args = runner.parse_args(["--card", "0", "--suite", suite])
    assert [job["suite"] for job in runner.build_jobs(args)] == expected


def test_default_memory_estimate_is_admitted_but_huge_shape_is_rejected():
    args = runner.parse_args(["--card", "0"])
    assert args.queries == 8192 and args.kv_length == 131072
    assert args.heads == 12 and args.iterations == 10
    assert args.memory_limit_mib == 4096 and args.timeout_seconds == 600
    admission = runner.memory_precheck(args, runner.build_jobs(args))
    assert admission["pass"]
    assert 512 < admission["estimated_peak_mib"] < 4096
    args.queries = 131072
    args.heads = 128
    assert not runner.memory_precheck(args, runner.build_jobs(args))["pass"]


def test_plan_only_creates_unique_summaries_without_launches(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only must not launch a worker")

    monkeypatch.setattr(runner, "run_child", forbidden)
    args = ["--card", "7", "--plan-only", "--output-dir", str(tmp_path)]
    assert runner.main(args) == 0
    assert runner.main(args) == 0
    summaries = _summaries(tmp_path)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["status"] == "incomplete"
        assert summary["all_passed"] is False
        assert summary["reason"] == "plan_only_no_hardware_tests"
        assert summary["physical_card"] == 7
        assert all(result["status"] == "incomplete" for result in summary["results"])


def test_plan_only_cli_never_imports_torch(tmp_path):
    source = (
        "import builtins, runpy, sys\n"
        "original = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in ('torch', 'torch_npu', 'cannbotdsl'):\n"
        "        raise RuntimeError('forbidden NPU import: ' + name)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        f"sys.argv = [{str(Path(runner.__file__))!r}, '--card', '0', '--plan-only', '--output-dir', {str(tmp_path)!r}]\n"
        f"runpy.run_path({str(Path(runner.__file__))!r}, run_name='__main__')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, timeout=5, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert _summaries(tmp_path)[0]["status"] == "incomplete"


def test_memory_rejection_never_launches_worker(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("rejected memory plan must not run")

    monkeypatch.setattr(runner, "run_child", forbidden)
    assert (
        runner.main(
            ["--card", "0", "--output-dir", str(tmp_path), "--memory-limit-mib", "1"]
        )
        == 1
    )
    summary = _summaries(tmp_path)[0]
    assert summary["status"] == "incomplete"
    assert summary["reason"] == "estimated_memory_exceeds_budget"
    assert not summary["all_passed"]


def test_optional_prolog_stops_on_missing_dependencies(tmp_path, monkeypatch):
    calls = []

    def fake_worker(command, suite, log_path, **kwargs):
        calls.append(suite)
        return {
            "suite": suite,
            "status": "failed" if suite == "prolog_preflight" else "passed",
            "reason": "missing_dependency",
            "returncode": 2 if suite == "prolog_preflight" else 0,
        }

    monkeypatch.setattr(runner, "run_child", fake_worker)
    assert (
        runner.main(["--card", "0", "--suite", "prolog", "--output-dir", str(tmp_path)])
        == 1
    )
    assert calls == ["prolog_preflight"]
    summary = _summaries(tmp_path)[0]
    assert summary["status"] == "failed"
    assert not summary["all_passed"]
    assert summary["results"][1]["status"] == "incomplete"


def test_all_stops_on_incomplete_worker(tmp_path, monkeypatch):
    calls = []

    def fake_worker(command, suite, log_path, **kwargs):
        calls.append(suite)
        return {
            "suite": suite,
            "status": "incomplete",
            "reason": "missing_suite_complete",
        }

    monkeypatch.setattr(runner, "run_child", fake_worker)
    assert runner.main(["--card", "0", "--output-dir", str(tmp_path)]) == 1
    assert calls == ["preflight"]
    assert _summaries(tmp_path)[0]["status"] == "incomplete"


def test_real_missing_dependency_preflight_stops_all(tmp_path, monkeypatch):
    workers = tmp_path / "workers"
    workers.mkdir()
    marker = tmp_path / "computation_started"
    _worker(
        workers,
        (
            "import sys\n"
            "from pathlib import Path\n"
            "suite = sys.argv[sys.argv.index('--suite') + 1]\n"
            "if suite == 'preflight':\n"
            "    print('required operator dependency is missing')\n"
            "    raise SystemExit(2)\n"
            f"Path({str(marker)!r}).touch()\n"
        ),
        name="qfm_probe.py",
    )
    real_build_jobs = runner.build_jobs
    monkeypatch.setattr(
        runner, "build_jobs", lambda args: real_build_jobs(args, workers)
    )
    output = tmp_path / "output"
    assert runner.main(["--card", "0", "--output-dir", str(output)]) == 1
    summary = _summaries(output)[0]
    assert summary["status"] == "failed"
    assert summary["results"][0]["returncode"] == 2
    assert not marker.exists()
    assert all(result["status"] == "incomplete" for result in summary["results"][1:])


def test_summary_is_passed_only_when_every_worker_passes(tmp_path, monkeypatch):
    calls = []

    def fake_worker(command, suite, log_path, **kwargs):
        calls.append(suite)
        return {
            "suite": suite,
            "status": "passed",
            "reason": "explicit_completion_and_zero_exit",
        }

    monkeypatch.setattr(runner, "run_child", fake_worker)
    assert runner.main(["--card", "0", "--output-dir", str(tmp_path)]) == 0
    summary = _summaries(tmp_path)[0]
    assert calls == ["preflight", "small", "graph", "dcp", "performance"]
    assert summary["status"] == "passed"
    assert summary["all_passed"]
    assert not summary["fused_prolog_validated"]
    assert summary["prolog_status"] == "prolog_not_requested"


@pytest.mark.parametrize("suite", ["inspect", "preflight"])
def test_inventory_success_is_not_native_validation(tmp_path, monkeypatch, suite):
    def fake_worker(command, suite, log_path, **kwargs):
        return {
            "suite": suite,
            "status": "passed",
            "reason": "explicit_completion_and_zero_exit",
        }

    monkeypatch.setattr(runner, "run_child", fake_worker)
    assert (
        runner.main(["--card", "0", "--suite", suite, "--output-dir", str(tmp_path)])
        == 0
    )
    summary = _summaries(tmp_path)[0]
    assert summary["status"] == "passed"
    assert not summary["native_computation_validated"]
    assert not summary["fused_prolog_validated"]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--card", "-1"],
        ["--card", "0", "--timeout-seconds", "0"],
        ["--card", "0", "--timeout-seconds", "nan"],
        ["--card", "0", "--memory-limit-mib", "inf"],
        ["--card", "0", "--queries", "2", "--kv-length", "1"],
        ["--card", "0", "--iterations", "0"],
    ],
)
def test_invalid_cli_is_rejected(args):
    with pytest.raises(SystemExit):
        runner.parse_args(args)
