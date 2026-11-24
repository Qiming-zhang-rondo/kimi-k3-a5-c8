#!/usr/bin/env python3
"""Signed GitHub push -> existing DeepSeek Harness 0.2.2-rc.1 session.

No polling, scheduling, headless session creation or guessed DSH CLI commands.
The launch URL/token stays in a private file outside source control.
"""
import argparse
from http.cookies import SimpleCookie
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit
import uuid

import github_push_notify as common

VERSION = "0.2.2-rc.1"
SOURCE_COMMIT = "66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db"
REQUEST_NAMESPACE = uuid.UUID("76383b7c-d93c-4f50-91f0-d92c1551b9cc")
FIELDS = {"version", "dsh_version", "dsh_base_url", "connect_url_file", "session_id", "repo", "branch",
          "secret_file", "state_dir", "port", "queue_timeout", "message_template", "ignore_commit_prefixes"}
NotificationError = common.NotificationError


def valid_session(value):
    if (not isinstance(value, str) or not value or len(value) > 512
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("session_id must be the exact nonempty existing DSH session ID")
    return value


def base_url(value):
    try:
        p = urlsplit(value)
        port = p.port
        if (not p.hostname or p.username or p.password or p.query or p.fragment
                or p.scheme not in ("http", "https")
                or (p.scheme == "http" and p.hostname not in ("127.0.0.1", "localhost", "::1"))
                or not re.fullmatch(r"/[A-Za-z0-9_./~-]*", p.path or "/")
                or any(c in (".", "..") for c in p.path.split("/")) or "//" in p.path):
            raise ValueError()
        host = p.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        if port is not None and port != (443 if p.scheme == "https" else 80):
            host += ":" + str(port)
        return urlunsplit((p.scheme, host, (p.path or "/").rstrip("/") + "/", "", ""))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("DSH URL must be HTTPS or loopback HTTP, without credentials/query; keep its mount prefix") from None


def read_launch(path):
    try:
        data = Path(path).read_bytes()
        if not 1 <= len(data) <= 65536:
            raise ValueError()
        p = urlsplit(data.decode("utf-8-sig").strip())
        query = parse_qs(p.query, keep_blank_values=True, strict_parsing=True)
        if p.fragment or set(query) != {"token"} or len(query["token"]) != 1:
            raise ValueError()
        token = query["token"][0]
        if not token or len(token) > 16384 or any(ord(c) < 32 or ord(c) == 127 for c in token):
            raise ValueError()
        path = p.path or "/"
        if path.endswith("/connect"):
            path = path[:-7]
        elif not path.endswith("/"):
            raise ValueError()
        root = base_url(urlunsplit((p.scheme, p.netloc, path, "", "")))
        return root, token
    except (ValueError, UnicodeError):
        raise ValueError("Launch URL file must contain one DSH launch root or /connect URL with exactly one token") from None


def backend_scope(config):
    return "dsh:" + VERSION + ":" + base_url(config["dsh_base_url"])


def scope_for_config(config):
    return json.dumps([backend_scope(config), config["repo"].lower(), config["branch"], config["session_id"]], separators=(",", ":"))


class _Deadline:
    """One deadline across auth, upgrade, HTTP admission and cleanup."""
    def __init__(self, timeout):
        if type(timeout) not in (int, float) or not 0 < timeout <= common.QUEUE_TIMEOUT:
            raise ValueError("Timeout must be positive and at most five seconds")
        self.end = time.monotonic() + timeout
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.closers = []

    def remaining(self):
        value = self.end - time.monotonic()
        if value <= 0 or self.cancelled.is_set():
            raise NotificationError("dsh_timeout")
        return value

    def own(self, closer):
        with self.lock:
            immediate = self.cancelled.is_set()
            if not immediate:
                self.closers.append(closer)
        if immediate:
            closer()

    def close(self):
        with self.lock:
            self.cancelled.set()
            closers, self.closers = self.closers, []
        for closer in reversed(closers):
            try:
                closer()
            except Exception:
                pass


def _bounded(timeout, operation):
    deadline = _Deadline(timeout)
    done = threading.Event()
    outcome = []

    def run():
        try:
            outcome.append((True, operation(deadline)))
        except NotificationError as error:
            outcome.append((False, error))
        except TimeoutError:
            outcome.append((False, NotificationError("dsh_timeout")))
        except Exception:
            # Do not expose token-bearing URLs, Cookie, or remote error bodies.
            code = "dsh_timeout" if time.monotonic() >= deadline.end else "dsh_transport_failed"
            outcome.append((False, NotificationError(code)))
        finally:
            deadline.close()
            done.set()

    threading.Thread(target=run, name="dsh-admission", daemon=True).start()
    if not done.wait(timeout):
        deadline.close()
        raise NotificationError("dsh_timeout")
    ok, value = outcome[0]
    if not ok:
        raise value
    return value


def _abort_http(connection):
    if connection.sock is not None:
        try:
            connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    connection.close()


class DshClient:
    def __init__(self, base_url, connect_url_file, session_id, websocket_factory=None,
                 notification_scope=None):
        self.base_url = globals()["base_url"](base_url)
        self.auth_file = Path(connect_url_file)
        self.session_id = valid_session(session_id)
        self.websocket_factory = websocket_factory
        self.scope = notification_scope or json.dumps(["dsh", self.base_url, self.session_id], separators=(",", ":"))

    def _http(self, deadline, method, path, body=None, cookie=None):
        p = urlsplit(self.base_url)
        connection_type = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(p.hostname, p.port, timeout=deadline.remaining())
        deadline.own(lambda: _abort_http(connection))
        headers = {"Accept": "application/json", "Connection": "close"}
        if cookie is not None:
            headers["Cookie"] = cookie
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            chunks, size = [], 0
            while True:
                deadline.remaining()
                if connection.sock is not None:
                    connection.sock.settimeout(deadline.remaining())
                chunk = response.read1(min(65536, common.MAX_BODY + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > common.MAX_BODY:
                    raise NotificationError("dsh_response_too_large")
            return response.status, response.getheaders(), b"".join(chunks)
        finally:
            connection.close()

    def _authenticate(self, deadline):
        # Check the optional dependency BEFORE /connect allocates a Host Peer.
        try:
            import websocket
        except ImportError:
            raise NotificationError("dsh_missing_websocket_client") from None
        try:
            root, token = read_launch(self.auth_file)
        except (OSError, ValueError):
            raise NotificationError("dsh_launch_file_invalid") from None
        if root != self.base_url:
            raise NotificationError("dsh_auth_origin_changed")
        p = urlsplit(root)
        connect_path = p.path + "connect?" + urlencode({"token": token, "interactions": "none"})
        status, headers, _ = self._http(deadline, "GET", connect_path)
        if status != 302:
            raise NotificationError("dsh_auth_rejected")
        locations = [v for k, v in headers if k.lower() == "location"]
        if len(locations) != 1:
            raise NotificationError("dsh_auth_redirect_invalid")
        mux = urljoin(root + "connect", locations[0])
        if mux != root + "api/remote.mux":
            raise NotificationError("dsh_auth_redirect_invalid")
        jar = SimpleCookie()
        for key, value in headers:
            if key.lower() == "set-cookie":
                jar.load(value)
        if not jar:
            raise NotificationError("dsh_auth_cookie_missing")
        # Only cookie name/value pairs cross to the API, never Set-Cookie attributes.
        cookie = "; ".join(key + "=" + morsel.coded_value for key, morsel in jar.items())
        if len(cookie) > 16384 or any(ord(c) < 32 or ord(c) == 127 for c in cookie):
            raise NotificationError("dsh_auth_cookie_invalid")
        factory = self.websocket_factory or websocket.create_connection
        wp = urlsplit(mux)
        ws_url = urlunsplit(("wss" if wp.scheme == "https" else "ws", wp.netloc, wp.path, "", ""))
        ws = factory(ws_url, timeout=deadline.remaining(), cookie=cookie,
                     suppress_origin=True, redirect_limit=0, http_no_proxy=["*"], enable_multithread=True)
        stop = threading.Event()

        def close_socket():
            stop.set()
            try:
                # Deadline expiry must abort immediately, never add a close
                # handshake's original socket timeout to the admission budget.
                remaining = deadline.end - time.monotonic()
                if ws.connected and remaining > 0:
                    ws.settimeout(min(0.05, remaining))
                    ws.send_close()
            except Exception:
                pass
            try:
                if ws.sock is not None:
                    ws.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            ws.shutdown()

        deadline.own(close_socket)
        # websocket-client redirect_limit=0 alone does not require a 101 result.
        if ws.getstatus() != 101:
            raise NotificationError("dsh_websocket_upgrade_failed")
        deadline.remaining()

        def receive_controls():
            try:
                while not stop.is_set():
                    opcode, _ = ws.recv_data_frame(control_frame=True)
                    if opcode == websocket.ABNF.OPCODE_CLOSE:
                        break
            except Exception:
                pass

        threading.Thread(target=receive_controls, name="dsh-ws-control", daemon=True).start()
        return cookie

    def _rpc(self, deadline, method, args):
        cookie = self._authenticate(deadline)
        rpc_id = str(uuid.uuid4())
        body = json.dumps({"type": "client-request", "rpcId": rpc_id, "method": method,
                           "payload": {"args": args}}, ensure_ascii=False).encode()
        path = urlsplit(self.base_url).path + "api/" + method
        status, _, data = self._http(deadline, "POST", path, body, cookie)
        if status != 200:
            raise NotificationError("dsh_rpc_http_failed")
        try:
            response = json.loads(data)
        except (ValueError, UnicodeError):
            raise NotificationError("dsh_rpc_invalid_response") from None
        if (not isinstance(response, dict) or response.get("type") != "server-response"
                or response.get("rpcId") != rpc_id or not isinstance(response.get("result"), dict)):
            raise NotificationError("dsh_rpc_invalid_response")
        result = response["result"]
        if result.get("ok") is not True:
            code = result.get("error", {}).get("code") if isinstance(result.get("error"), dict) else None
            allowed = {"session/not-found": "dsh_session_not_found", "session/writer-held": "dsh_session_writer_held",
                       "session/agent-busy": "dsh_session_busy"}
            raise NotificationError(allowed.get(code, "dsh_rpc_rejected"))
        return result.get("value")

    def queue(self, message, sha, delivery, timeout=5):
        del delivery  # Commit-scoped identity also survives a new GitHub delivery ID.
        request_id = str(uuid.uuid5(REQUEST_NAMESPACE, self.scope + "\n" + sha))

        def submit(deadline):
            value = self._rpc(deadline, "session/prompt", {"request": {
                "sessionId": self.session_id, "requestId": request_id, "mode": "queue",
                "content": [{"type": "text", "text": message}], "clientTimeZone": "Asia/Shanghai"}})
            if not isinstance(value, dict) or value.get("accepted") is not True:
                raise NotificationError("dsh_prompt_not_accepted")
        return _bounded(timeout, submit)

    def check(self, timeout=5):
        value = _bounded(timeout, lambda d: self._rpc(d, "session/state", {"request": {"sessionId": self.session_id}}))
        if (not isinstance(value, dict) or type(value.get("active")) is not bool
                or type(value.get("running")) is not bool or type(value.get("cursor")) is not int
                or not isinstance(value.get("pendingInput"), list)
                or value.get("queue") is not None and not isinstance(value.get("queue"), list)):
            raise NotificationError("dsh_state_invalid")
        return {"status": "CONFIG_READY", "session_id": self.session_id, "active": value["active"],
                "running": value["running"], "configured_version": VERSION, "runtime_version_verified": False,
                "chat_delivery_tested": False, "model_message_sent": False}

    def sessions(self, timeout=5):
        value = _bounded(timeout, lambda d: self._rpc(d, "session/list", {"_request": {}}))
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise NotificationError("dsh_session_list_invalid")
        return {"items": [{k: item[k] for k in ("sessionId", "running", "agentAvailable", "cwd") if k in item}
                          for item in value["items"] if isinstance(item, dict)], "model_message_sent": False}


class DshReceiver(common.Receiver):
    def __init__(self, config):
        self.client = DshClient(config["dsh_base_url"], config["connect_url_file"], config["session_id"],
                                notification_scope=scope_for_config(config))
        super().__init__(thread=config["session_id"], secret=Path(config["secret_file"]).read_bytes(),
                         state_dir=config["state_dir"], codex="unused-dsh-backend", repo=config["repo"],
                         branch=config["branch"], notifier=self.client.queue, backend_scope=backend_scope(config),
                         queue_timeout=config.get("queue_timeout", 5), message_template=config.get("message_template"),
                         ignore_commit_prefixes=config.get("ignore_commit_prefixes", ["[codex-review]"]))


def receiver_from_config(config):
    return DshReceiver(config)


def read_status(config):
    return common.read_status({**config, "_scope": scope_for_config(config)})


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"version", "dsh_version", "dsh_base_url", "connect_url_file", "session_id", "repo", "branch", "secret_file", "state_dir"}
    if (not isinstance(config, dict) or set(config) - FIELDS or required - set(config)
            or type(config["version"]) is not int or config["version"] != 1 or config["dsh_version"] != VERSION):
        raise ValueError("DSH configuration requires version 1, dsh_version 0.2.2-rc.1 and all routing/path fields")
    config["dsh_base_url"] = base_url(config["dsh_base_url"])
    config["session_id"] = valid_session(config["session_id"])
    config["repo"], config["branch"] = common.valid_repo(config["repo"]), common.valid_branch(config["branch"])
    for key in ("connect_url_file", "secret_file", "state_dir"):
        config[key] = common._path(config[key], path.parent)
    config.setdefault("port", 8766)
    config.setdefault("queue_timeout", 5)
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if type(config["queue_timeout"]) not in (int, float) or not 0 < config["queue_timeout"] <= 5:
        raise ValueError("queue_timeout must be positive and at most five seconds")
    config["message_template"] = common.valid_template(config.get("message_template", common.DEFAULT_TEMPLATE))
    config["ignore_commit_prefixes"] = common.valid_prefixes(config.get("ignore_commit_prefixes", ["[codex-review]"]))
    return config


def init_config(path, repo, branch, session_id, connect_url_file, port=8766):
    path = Path(path).expanduser().resolve()
    source = Path(__file__).resolve().parent.parent
    if path.is_relative_to(source) or any((p / ".git").exists() for p in (path.parent, *path.parents)):
        raise ValueError("Keep configuration and credentials outside source/Git repositories")
    root, token = read_launch(connect_url_file)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    config = {"version": 1, "dsh_version": VERSION, "dsh_base_url": root, "session_id": valid_session(session_id),
              "repo": common.valid_repo(repo), "branch": common.valid_branch(branch),
              "connect_url_file": path.stem + ".connect-url", "secret_file": path.stem + ".secret",
              "state_dir": path.stem + ".state", "port": port, "queue_timeout": 5,
              "message_template": common.DEFAULT_TEMPLATE, "ignore_commit_prefixes": ["[codex-review]"]}
    targets = [path, path.parent / config["connect_url_file"], path.parent / config["secret_file"]]
    if any(p.exists() for p in targets):
        raise ValueError("Refusing to overwrite existing configuration or credentials")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    created = []
    try:
        contents = [json.dumps(config, ensure_ascii=False, indent=2) + "\n", root + "?" + urlencode({"token": token}), secrets.token_hex(32)]
        for target, content in zip(targets, contents):
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created.append(target)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
    except OSError:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create private routing configuration; no network/model request")
    dest = init.add_mutually_exclusive_group(required=True)
    dest.add_argument("--config", type=Path)
    dest.add_argument("--dir", type=Path)
    init.add_argument("--repo", required=True)
    init.add_argument("--branch", required=True)
    init.add_argument("--session", required=True)
    init.add_argument("--connect-url-file", type=Path, required=True)
    init.add_argument("--port", type=int, default=8766)
    for command in ("serve", "check", "test", "status"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True, type=Path)
        if command == "test":
            p.add_argument("--url", help="Optional public HTTPS /github; sends only a signed ping")
    listing = sub.add_parser("sessions", help="Read existing session IDs without creating/resuming a session")
    listing.add_argument("--connect-url-file", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            path = init_config(args.config or args.dir / "config.json", args.repo, args.branch,
                               args.session, args.connect_url_file, args.port)
            result, code = {"status": "initialized", "config": str(path)}, 0
        elif args.command == "sessions":
            root, _ = read_launch(args.connect_url_file)
            result, code = DshClient(root, args.connect_url_file, "read-only-session-list").sessions(), 0
        else:
            config = load_config(args.config)
            if args.command == "serve":
                # Fail before opening the webhook port if the read-only DSH check fails.
                DshClient(config["dsh_base_url"], config["connect_url_file"], config["session_id"]).check(config["queue_timeout"])
                server = common.make_server(receiver_from_config(config), config["port"])
                print(json.dumps({"status": "listening", "address": "127.0.0.1", "port": config["port"],
                                  "backend": "dsh", "configured_version": VERSION}), flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    server.server_close()
                return 0
            if args.command == "check":
                if len(Path(config["secret_file"]).read_bytes()) < 32:
                    raise ValueError("Webhook secret must contain at least 32 bytes")
                result, code = DshClient(config["dsh_base_url"], config["connect_url_file"], config["session_id"]).check(config["queue_timeout"]), 0
            elif args.command == "test":
                result, code = common.ping_config(config, args.url)
            else:
                result, code = read_status(config), 0
        print(json.dumps(result, ensure_ascii=False))
        return code
    except (NotificationError, OSError, ValueError, sqlite3.Error, http.client.HTTPException) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__,
                          "reason": error.code if isinstance(error, NotificationError) else
                          str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else "operation_failed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
