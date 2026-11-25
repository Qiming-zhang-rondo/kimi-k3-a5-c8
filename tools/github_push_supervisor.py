#!/usr/bin/env python3
"""Event-driven local receiver + Cloudflare tunnel + owned GitHub webhook.

No GitHub polling, scheduled jobs, or synthetic chat messages. Cloudflare output
is retained only in a private local log; credentials and URLs are never printed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading

import github_push_notify as notify


QUICK_URL = re.compile(r"https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com(?=$|[\s\"'|])")


class SupervisorError(Exception):
    """Only a credential-free machine-readable error code may leave helpers."""


class ShutdownRequested(BaseException):
    pass


def extract_endpoint(line):
    match = QUICK_URL.search(line)
    return match.group(0) + "/github" if match else None


def private_json(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def health(runtime_dir, branch, status, *, hook_id=None, error=None):
    state = dict(status=status, timestamp=datetime.now(timezone.utc).isoformat(), branch=branch)
    if hook_id is not None:
        state["hook_id"] = hook_id
    if error is not None:
        state["error"] = error
    private_json(Path(runtime_dir) / "health.json", state)
    print(json.dumps(state), flush=True)


def credential_token(runner=subprocess.run):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="false", SSH_ASKPASS="false")
    try:
        result = runner(["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True,
            timeout=10, env=environment, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        raise SupervisorError("credential_helper_failed") from None
    if result.returncode:
        raise SupervisorError("credential_helper_failed")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = fields.get("password", "")
    if not token or any(character in token for character in "\r\n\0"):
        raise SupervisorError("credential_missing")
    return token


def curl_quote(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'


class GitHubApi:
    def __init__(self, token, runner=subprocess.run):
        if not token or any(character in token for character in "\r\n\0"):
            raise SupervisorError("credential_invalid")
        self.token, self.runner = token, runner

    def request(self, method, path, payload=None):
        if method not in {"GET", "POST", "PATCH"} or not re.fullmatch(r"/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/hooks(?:/[1-9][0-9]*)?)?", path):
            raise SupervisorError("api_request_invalid")
        options = ["silent", "show-error", "connect-timeout = 10", "max-time = 30",
            'proto = "=https"', 'tlsv1.2', "request = " + curl_quote(method),
            "url = " + curl_quote("https://api.github.com" + path),
            "header = " + curl_quote("Authorization: Bearer " + self.token),
            'header = "Accept: application/vnd.github+json"',
            'header = "X-GitHub-Api-Version: 2022-11-28"',
            'write-out = "\\n%{http_code}"']
        if payload is not None:
            options += ['header = "Content-Type: application/json"',
                "data = " + curl_quote(json.dumps(payload, separators=(",", ":")))]
        try:
            result = self.runner(["curl", "--disable", "--config", "-"], input="\n".join(options) + "\n",
                text=True, capture_output=True, timeout=35, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            raise SupervisorError("github_transport_failed") from None
        if result.returncode:
            raise SupervisorError("github_transport_failed")
        try:
            body, status = result.stdout.rsplit("\n", 1)
            code, data = int(status), json.loads(body)
        except (ValueError, TypeError):
            raise SupervisorError("github_response_invalid") from None
        if not 200 <= code < 300:
            raise SupervisorError("github_http_" + str(code))
        return data


def webhook_payload(endpoint, secret):
    if not isinstance(endpoint, str) or not re.fullmatch(r"https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com/github", endpoint):
        raise SupervisorError("tunnel_endpoint_invalid")
    if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
        raise SupervisorError("webhook_secret_invalid")
    return dict(name="web", active=True, events=["push"], config=dict(
        url=endpoint, content_type="json", insecure_ssl="0", secret=secret))


def register_hook(config, endpoint, runtime_dir, api):
    """Only the hook ID recorded by this service may ever be patched."""
    repo, thread = config["repo"], config["thread"]
    state_path = Path(runtime_dir) / "hook_state.json"
    try:
        secret = Path(config["secret_file"]).read_bytes().decode("utf-8")
        state = json.loads(state_path.read_text()) if state_path.exists() else None
    except (OSError, ValueError, UnicodeError):
        raise SupervisorError("private_state_invalid") from None
    payload = webhook_payload(endpoint, secret)
    base = "/repos/" + repo
    if state is not None:
        if not isinstance(state, dict):
            raise SupervisorError("owned_hook_state_invalid")
        hook_id = state.get("hook_id")
        if (state.get("version") != 1 or state.get("repo") != repo or state.get("thread") != thread
                or type(hook_id) is not int or hook_id <= 0):
            raise SupervisorError("owned_hook_state_invalid")
        existing = api.request("GET", base + "/hooks/" + str(hook_id))
        if not isinstance(existing, dict) or existing.get("id") != hook_id or existing.get("name") != "web":
            raise SupervisorError("owned_hook_mismatch")
        response = api.request("PATCH", base + "/hooks/" + str(hook_id), payload)
    else:
        repository = api.request("GET", base)
        if not isinstance(repository, dict) or repository.get("permissions", {}).get("admin") is not True:
            raise SupervisorError("repository_admin_required")
        response = api.request("POST", base + "/hooks", payload)
        hook_id = response.get("id") if isinstance(response, dict) else None
        if type(hook_id) is not int or hook_id <= 0:
            raise SupervisorError("hook_creation_response_invalid")
    if not isinstance(response, dict) or response.get("id") != hook_id:
        raise SupervisorError("owned_hook_response_invalid")
    private_json(state_path, dict(version=1, repo=repo, thread=thread, hook_id=hook_id))
    return hook_id


def supervise(config, cloudflared, runtime_dir):
    runtime_dir = Path(runtime_dir).expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(runtime_dir, 0o700)
    server = child = log = None
    thread = None
    branch, hook_id = config["branch"], None
    saved_signals = {}

    def stop(_signum, _frame):
        raise ShutdownRequested()

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            saved_signals[signum] = signal.signal(signum, stop)
        server = notify.make_server(notify.receiver_from_config(config), config["port"])
        thread = threading.Thread(target=server.serve_forever, name="github-push-receiver", daemon=True)
        thread.start()
        health(runtime_dir, branch, "STARTING")
        descriptor = os.open(runtime_dir / "cloudflared.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(descriptor, 0o600)
        log = os.fdopen(descriptor, "w", encoding="utf-8")
        child = subprocess.Popen([str(cloudflared), "tunnel", "--no-autoupdate", "--protocol", "http2",
            "--url", "http://127.0.0.1:" + str(server.server_address[1])],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        registered_endpoint = None
        api = None
        for line in child.stdout:
            log.write(line)
            log.flush()
            endpoint = extract_endpoint(line)
            if endpoint and endpoint != registered_endpoint:
                if api is None:
                    api = GitHubApi(credential_token())
                hook_id = register_hook(config, endpoint, runtime_dir, api)
                registered_endpoint = endpoint
                health(runtime_dir, branch, "REGISTERED", hook_id=hook_id)
        child.wait()
        raise SupervisorError("tunnel_exited")
    except ShutdownRequested:
        health(runtime_dir, branch, "STOPPED", hook_id=hook_id)
        return 0
    except SupervisorError as error:
        health(runtime_dir, branch, "FAILED", hook_id=hook_id, error=str(error))
        return 1
    except Exception:
        # Exception strings may contain a subprocess's URL or credential input.
        health(runtime_dir, branch, "FAILED", hook_id=hook_id, error="supervisor_failed")
        return 1
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child.stdout is not None:
                child.stdout.close()
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        if log is not None:
            log.close()
        for signum, previous in saved_signals.items():
            signal.signal(signum, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cloudflared", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = notify.load_config(args.config)
        cloudflared = args.cloudflared.expanduser().resolve(strict=True)
        if not os.access(cloudflared, os.X_OK):
            raise SupervisorError("cloudflared_not_executable")
        return supervise(config, cloudflared, args.runtime_dir)
    except Exception:
        print(json.dumps(dict(status="FAILED", error="startup_configuration_invalid")), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
