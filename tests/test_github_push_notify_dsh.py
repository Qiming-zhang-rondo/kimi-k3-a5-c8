"""Pinned DSH wire-contract integration tests; no DSH/model installation needed.

The fixture implements real HTTP and the RFC6455 upgrade/control-frame lifecycle.
It is deliberately not a fake notifier: signed GitHub pushes cross both local
HTTP boundaries and exercise the production DSH HTTP/WebSocket client.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import base64
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.parse import parse_qs, urlsplit
from types import SimpleNamespace
import uuid

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import github_push_notify as core
spec = importlib.util.spec_from_file_location("github_push_notify_dsh", TOOLS / "github_push_notify_dsh.py")
notify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notify)


def _read_exact(connection, length):
    result = b""
    while len(result) < length:
        part = connection.recv(length - len(result))
        if not part:
            raise EOFError()
        result += part
    return result


class DshHostFixture:
    """Loopback-only subset of rc.1 Host HTTP Remote and lifecycle WebSocket."""

    def __init__(self, mount=""):
        self.mount = mount
        self.token = "fixture-launch-secret-DO-NOT-LOG"
        self.cookie = "dsh_fixture=signed-cookie-DO-NOT-LOG"
        self.session = "existing-session-with-non-uuid-id"
        self.requests = []
        self.auth_requests = []
        self.ws_headers = []
        self.ws_opened = threading.Event()
        self.ws_pong = threading.Event()
        self.ws_closed = threading.Event()
        self.seen_request_ids = set()
        self.admissions = 0
        self.prompt_reply = "success"
        # rc.1 browser-auth returns this relative Location, not an API URL body.
        self.connect_redirect = "./api/remote.mux"
        self.authorized = True
        self.connections = []
        self.stopping = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def reply(self, status, body=b"", **headers):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                for key, value in headers.items():
                    self.send_header(key.replace("_", "-"), value)
                self.end_headers()
                if body:
                    try:
                        self.wfile.write(body)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path == fixture.mount + "/connect":
                    fixture.auth_requests.append(parse_qs(parsed.query))
                    if parse_qs(parsed.query).get("token") != [fixture.token]:
                        return self.reply(401)
                    return self.reply(302, Location=fixture.connect_redirect,
                                      Set_Cookie=fixture.cookie + "; HttpOnly; Path=/; SameSite=Lax")
                if parsed.path != fixture.mount + "/api/remote.mux" or self.headers.get("Upgrade", "").lower() != "websocket":
                    return self.reply(404)
                fixture.ws_headers.append(dict(self.headers))
                if self.headers.get("Cookie") != fixture.cookie or not fixture.authorized:
                    return self.reply(401)
                key = self.headers.get("Sec-WebSocket-Key", "")
                accept = base64.b64encode(hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                self.wfile.flush()
                fixture.connections.append(self.connection)
                fixture.ws_opened.set()
                self.connection.settimeout(2)
                # A real control-frame ping makes the client's read pump observable.
                self.connection.sendall(b"\x89\x07fixture")
                try:
                    while True:
                        first, second = _read_exact(self.connection, 2)
                        length = second & 127
                        if length == 126:
                            length = struct.unpack("!H", _read_exact(self.connection, 2))[0]
                        elif length == 127:
                            length = struct.unpack("!Q", _read_exact(self.connection, 8))[0]
                        mask = _read_exact(self.connection, 4) if second & 128 else b""
                        data = _read_exact(self.connection, length)
                        if mask:
                            data = bytes(value ^ mask[i % 4] for i, value in enumerate(data))
                        opcode = first & 15
                        if opcode == 10 and data == b"fixture":
                            fixture.ws_pong.set()
                        elif opcode == 8:
                            self.connection.sendall(bytes([0x88, len(data)]) + data)
                            fixture.ws_closed.set()
                            return
                except (OSError, EOFError):
                    fixture.ws_closed.set()
                finally:
                    self.close_connection = True

            def do_POST(self):
                if self.headers.get("Cookie") != fixture.cookie or not fixture.authorized:
                    return self.reply(401)
                try:
                    document = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                except (ValueError, TypeError):
                    return self.reply(400)
                fixture.requests.append((self.path, dict(self.headers), document))
                method = document.get("method")
                if self.path != fixture.mount + "/api/" + str(method):
                    return self.reply(404)
                rpc_id = document.get("rpcId")
                if method == "session/state":
                    request = document.get("payload", {}).get("args", {}).get("request", {})
                    if request.get("sessionId") != fixture.session:
                        response = {"type": "server-response", "rpcId": rpc_id,
                                    "result": {"ok": False, "error": {"code": "session/not-found"}}}
                        return self.reply(200, json.dumps(response).encode(), Content_Type="application/json")
                    value = {"cursor": 0, "active": False, "running": False, "queue": None,
                             "pendingInput": []}
                elif method == "session/list":
                    value = {"items": [{"sessionId": fixture.session, "agentAvailable": True,
                                        "running": False, "blank": False, "updatedAt": 1,
                                        "private_history": "must not leak"}]}
                elif method == "session/prompt":
                    request = document.get("payload", {}).get("args", {}).get("request", {})
                    if request.get("sessionId") != fixture.session:
                        response = {"type": "server-response", "rpcId": rpc_id,
                                    "result": {"ok": False, "error": {"code": "session/not-found"}}}
                        return self.reply(200, json.dumps(response).encode(), Content_Type="application/json")
                    request_id = request.get("requestId")
                    if request_id not in fixture.seen_request_ids:
                        fixture.seen_request_ids.add(request_id)
                        fixture.admissions += 1
                    # Keep the RPC active long enough to verify the WS pump.
                    fixture.ws_pong.wait(.5)
                    if fixture.prompt_reply == "http_error":
                        return self.reply(503)
                    if fixture.prompt_reply == "malformed":
                        return self.reply(200, b"not-json", Content_Type="application/json")
                    if fixture.prompt_reply == "oversized":
                        return self.reply(200, b" " * (core.MAX_BODY + 1), Content_Type="application/json")
                    if fixture.prompt_reply == "redirect":
                        return self.reply(302, Location="https://untrusted.invalid/api/session/prompt")
                    if fixture.prompt_reply == "slow_body":
                        self.send_response(200)
                        self.send_header("Content-Length", "100")
                        self.end_headers()
                        try:
                            for _ in range(100):
                                self.wfile.write(b" ")
                                self.wfile.flush()
                                if fixture.stopping.wait(.04):
                                    break
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        self.close_connection = True
                        return
                    value = {"accepted": fixture.prompt_reply != "rejected"}
                else:
                    return self.reply(404)
                response = {"type": "server-response", "rpcId": rpc_id,
                            "result": {"ok": True, "value": value}}
                if fixture.prompt_reply == "wrong_rpc" and method == "session/prompt":
                    response["rpcId"] = "wrong-rpc-id"
                if fixture.prompt_reply in ("remote_error", "sensitive_error") and method == "session/prompt":
                    response["result"] = {"ok": False, "error": {"code": "gateway/unavailable",
                        "message": fixture.token + " " + fixture.cookie if fixture.prompt_reply == "sensitive_error"
                        else "temporary admission failure"}}
                self.reply(200, json.dumps(response).encode(), Content_Type="application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base_url = "http://127.0.0.1:" + str(self.server.server_port) + mount + "/"
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stopping.set()
        for connection in self.connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


class DshReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import websocket
        except ImportError:
            raise unittest.SkipTest("Install tools/requirements-dsh-notify.txt for DSH transport integration tests")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dsh receiver integration ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.secret = b"signed-github-secret-at-least-32-bytes"
        self.payload = {"repository": {"full_name": "example/project"}, "ref": "refs/heads/main",
                        "after": "a" * 40, "head_commit": {"message": "result data"}}

    def setup_config(self, host, session=None):
        launch = self.root / "dsh-connect-url.secret"
        launch.write_text(host.base_url + "connect?token=" + host.token)
        launch.chmod(0o600)
        secret = self.root / "github.secret"
        secret.write_bytes(self.secret)
        return {"version": 1, "dsh_version": "0.2.2-rc.1", "dsh_base_url": host.base_url,
                "connect_url_file": str(launch), "session_id": session or host.session,
                "repo": "example/project", "branch": "main", "secret_file": str(secret),
                "state_dir": str(self.root / "state"), "port": 8766, "queue_timeout": 5}

    def signed(self, delivery=None):
        body = json.dumps(self.payload).encode()
        return {"X-GitHub-Event": "push", "X-GitHub-Delivery": delivery or str(uuid.uuid4()),
                "X-Hub-Signature-256": "sha256=" + hmac.new(self.secret, body, hashlib.sha256).hexdigest()}, body

    def prompt_requests(self, host):
        return [document for _, _, document in host.requests if document.get("method") == "session/prompt"]

    @contextmanager
    def serving(self, receiver):
        server = core.make_server(receiver, 0)
        worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            worker.join(2)

    def send_push(self, server, secret=None):
        headers, body = self.signed()
        if secret is not None:
            headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
        connection = http.client.HTTPConnection(*server.server_address, timeout=7)
        try:
            connection.request("POST", "/github", body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())["status"]
        finally:
            connection.close()

    def test_signed_push_uses_exact_rc1_envelope_and_websocket_lifecycle(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            receiver = notify.receiver_from_config(config)
            with self.serving(receiver) as server:
                self.assertEqual(self.send_push(server), (202, "queued"))
            self.assertTrue(host.ws_opened.is_set())
            self.assertTrue(host.ws_pong.is_set(), "lifecycle WebSocket must read ping frames")
            self.assertTrue(host.ws_closed.wait(1), "lifecycle WebSocket must close after receipt")
            requests = self.prompt_requests(host)
            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertEqual(request["type"], "client-request")
            self.assertEqual(request["method"], "session/prompt")
            uuid.UUID(request["rpcId"])
            prompt = request["payload"]["args"]["request"]
            self.assertEqual(prompt["sessionId"], host.session)
            self.assertEqual(prompt["mode"], "queue")
            uuid.UUID(prompt["requestId"])
            self.assertEqual(prompt["content"][0]["type"], "text")
            self.assertIn("a" * 40, prompt["content"][0]["text"])
            self.assertEqual(host.requests[-1][0], "/api/session/prompt")
            self.assertEqual(host.auth_requests[0].get("interactions"), ["none"])
            self.assertEqual(notify.read_status(config)["accepted"], 1)

    def test_success_dedupes_across_receiver_restart_and_delivery_ids(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            receiver = notify.receiver_from_config(config)
            delivery = str(uuid.uuid4())
            self.assertEqual(receiver.handle(*self.signed(delivery)), (202, "queued"))
            restored = notify.receiver_from_config(config)
            self.assertEqual(restored.handle(*self.signed(delivery)), (200, "duplicate"))
            self.assertEqual(restored.handle(*self.signed()), (200, "duplicate"))
            self.assertEqual(host.admissions, 1)
            self.assertEqual(len(self.prompt_requests(host)), 1)

    def test_ambiguous_receipt_retries_same_request_id_but_new_rpc_id(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            host.prompt_reply = "wrong_rpc"
            receiver = notify.receiver_from_config(config)
            self.assertEqual(receiver.handle(*self.signed())[0], 503)
            self.assertEqual(notify.read_status(config)["accepted"], 0)
            host.prompt_reply = "success"
            restored = notify.receiver_from_config(config)
            self.assertEqual(restored.handle(*self.signed()), (202, "queued"))
            requests = self.prompt_requests(host)
            self.assertEqual(len(requests), 2)
            first, second = [request["payload"]["args"]["request"]["requestId"] for request in requests]
            self.assertEqual(first, second)
            self.assertNotEqual(requests[0]["rpcId"], requests[1]["rpcId"])
            self.assertEqual(host.admissions, 1)

    def test_non_acceptance_never_marks_delivery_success(self):
        for failure in ("rejected", "remote_error", "http_error", "malformed"):
            with self.subTest(failure=failure), DshHostFixture() as host:
                config = self.setup_config(host)
                # Separate state for each failure so no successful scope can mask it.
                config["state_dir"] = str(self.root / failure)
                host.prompt_reply = failure
                receiver = notify.receiver_from_config(config)
                self.assertEqual(receiver.handle(*self.signed())[0], 503)
                self.assertEqual(notify.read_status(config)["accepted"], 0)

    def test_invalid_github_signature_never_authenticates_to_dsh(self):
        with DshHostFixture() as host:
            receiver = notify.receiver_from_config(self.setup_config(host))
            headers, body = self.signed()
            self.assertEqual(receiver.handle(headers, body + b" ")[0], 401)
            self.assertEqual(host.auth_requests, [])
            self.assertEqual(host.requests, [])

    def test_host_is_part_of_scope_and_session_is_not_uuid_validated(self):
        with DshHostFixture() as first_host, DshHostFixture() as second_host:
            first_config = self.setup_config(first_host)
            # Materialize first receiver before the second helper updates its file.
            first = notify.receiver_from_config(first_config)
            first_launch = Path(first_config["connect_url_file"])
            saved_first_launch = self.root / "first.secret"
            saved_first_launch.write_text(first_launch.read_text())
            first_config["connect_url_file"] = str(saved_first_launch)
            first = notify.receiver_from_config(first_config)
            second_config = self.setup_config(second_host)
            second = notify.receiver_from_config(second_config)
            delivery = str(uuid.uuid4())
            self.assertEqual(first.handle(*self.signed(delivery)), (202, "queued"))
            self.assertEqual(second.handle(*self.signed(delivery)), (202, "queued"))
            self.assertEqual(first_host.admissions, 1)
            self.assertEqual(second_host.admissions, 1)

    def test_missing_session_does_not_call_create_or_accept_prompt(self):
        with DshHostFixture() as host:
            config = self.setup_config(host, session="does-not-exist")
            receiver = notify.receiver_from_config(config)
            self.assertEqual(receiver.handle(*self.signed())[0], 503)
            self.assertEqual(notify.read_status(config)["accepted"], 0)
            self.assertEqual(host.admissions, 0)
            self.assertFalse(any("create" in request[2].get("method", "") for request in host.requests))

    def test_cross_origin_connect_redirect_is_not_followed(self):
        with DshHostFixture() as host:
            host.connect_redirect = "https://untrusted.invalid/api/remote.mux"
            receiver = notify.receiver_from_config(self.setup_config(host))
            self.assertEqual(receiver.handle(*self.signed())[0], 503)
            self.assertEqual(host.ws_headers, [])
            self.assertEqual(host.requests, [])

    def test_mount_prefix_is_preserved_for_auth_websocket_and_rpc(self):
        with DshHostFixture(mount="/dsh") as host:
            receiver = notify.receiver_from_config(self.setup_config(host))
            self.assertEqual(receiver.handle(*self.signed()), (202, "queued"))
            self.assertTrue(host.ws_opened.is_set())
            self.assertEqual(self.prompt_requests(host)[0]["method"], "session/prompt")
            self.assertEqual(host.requests[-1][0], "/dsh/api/session/prompt")

    def test_check_reads_existing_session_state_without_prompt_or_create(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            client = notify.DshClient(host.base_url, config["connect_url_file"], host.session)
            client.check()
            self.assertEqual(len(host.requests), 1)
            path, _, request = host.requests[0]
            self.assertEqual(path, "/api/session/state")
            self.assertEqual(request["method"], "session/state")
            self.assertEqual(request["payload"]["args"]["request"], {"sessionId": host.session})
            self.assertEqual(self.prompt_requests(host), [])
            self.assertEqual(host.admissions, 0)

    def test_check_missing_session_is_rejected_without_prompt_or_create(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            client = notify.DshClient(host.base_url, config["connect_url_file"], "does-not-exist")
            with self.assertRaises(Exception):
                client.check()
            self.assertEqual([request[2]["method"] for request in host.requests], ["session/state"])
            self.assertEqual(host.admissions, 0)

    def test_remote_error_secrets_are_not_persisted_or_logged(self):
        with DshHostFixture() as host:
            host.prompt_reply = "sensitive_error"
            config = self.setup_config(host)
            receiver = notify.receiver_from_config(config)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(receiver.handle(*self.signed())[0], 503)
                state = notify.read_status(config)
            persisted = (Path(config["state_dir"]) / "deliveries.sqlite3").read_bytes()
            for secret in (host.token, host.cookie):
                self.assertNotIn(secret, output.getvalue())
                self.assertNotIn(secret, json.dumps(state))
                self.assertNotIn(secret.encode(), persisted)

    def test_init_keeps_launch_secret_external_and_rejects_overwrite(self):
        with DshHostFixture() as host:
            source = self.setup_config(host)
            target = self.root / "settings" / "config.json"
            notify.init_config(target, "example/project", "main", host.session, source["connect_url_file"])
            config = notify.load_config(target)
            self.assertEqual(config["dsh_version"], "0.2.2-rc.1")
            self.assertEqual(config["dsh_base_url"], host.base_url)
            self.assertEqual(config["session_id"], host.session)
            self.assertEqual(config["port"], 8766)
            self.assertNotIn(host.token, target.read_text())
            self.assertEqual(host.auth_requests, [])
            if sys.platform != "win32":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)
                self.assertEqual(Path(config["secret_file"]).stat().st_mode & 0o777, 0o600)
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                notify.init_config(target, "example/project", "main", host.session, source["connect_url_file"])
            self.assertEqual(target.read_bytes(), before)

    def test_signed_ping_does_not_contact_dsh_or_send_model_message(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            receiver = notify.receiver_from_config(config)
            with self.serving(receiver) as server:
                ping_config = {**config, "port": server.server_port}
                result, code = core.ping_config(ping_config)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "PONG")
            self.assertFalse(result["model_message_sent"])
            self.assertEqual(host.auth_requests, [])
            self.assertEqual(host.requests, [])

    def test_session_list_uses_pinned_request_name_and_returns_only_safe_fields(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            client = notify.DshClient(host.base_url, config["connect_url_file"], host.session)
            result = client.sessions()
            self.assertFalse(result["model_message_sent"])
            self.assertEqual(result["items"], [{"sessionId": host.session, "running": False,
                                                "agentAvailable": True}])
            self.assertEqual(len(host.requests), 1)
            path, _, request = host.requests[0]
            self.assertEqual(path, "/api/session/list")
            self.assertEqual(request["payload"]["args"], {"_request": {}})
            self.assertEqual(host.admissions, 0)

    def test_rpc_response_body_limit_and_redirect_fail_without_following(self):
        for failure, expected in (("oversized", "dsh_response_too_large"), ("redirect", "dsh_rpc_http_failed")):
            with self.subTest(failure=failure), DshHostFixture() as host:
                config = self.setup_config(host)
                config["state_dir"] = str(self.root / failure)
                host.prompt_reply = failure
                receiver = notify.receiver_from_config(config)
                self.assertEqual(receiver.handle(*self.signed())[0], 503)
                state = notify.read_status(config)
                self.assertEqual(state["accepted"], 0)
                self.assertEqual(state["last_delivery"]["error"], expected)
                self.assertEqual(len(self.prompt_requests(host)), 1)

    def test_one_total_deadline_stops_slow_drip_then_retry_keeps_request_id(self):
        with DshHostFixture() as host:
            config = self.setup_config(host)
            config["queue_timeout"] = .2
            host.prompt_reply = "slow_body"
            receiver = notify.receiver_from_config(config)
            start = time.monotonic()
            self.assertEqual(receiver.handle(*self.signed())[0], 503)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, .7, "deadline must cover the whole attempt, not reset per body byte")
            state = notify.read_status(config)
            self.assertEqual(state["accepted"], 0)
            self.assertEqual(state["last_delivery"]["error"], "dsh_timeout")
            self.assertTrue(host.ws_closed.wait(1))
            host.prompt_reply = "success"
            self.assertEqual(receiver.handle(*self.signed()), (202, "queued"))
            requests = self.prompt_requests(host)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["payload"]["args"]["request"]["requestId"],
                             requests[1]["payload"]["args"]["request"]["requestId"])
            self.assertEqual(host.admissions, 1)

    def test_real_cli_init_check_serve_ping_status_and_signed_push(self):
        with DshHostFixture() as host:
            source = self.setup_config(host)
            target = self.root / "cli-settings" / "config.json"
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [sys.executable, str(TOOLS / "github_push_notify_dsh.py")]

            def cli(*arguments):
                completed = subprocess.run(command + list(map(str, arguments)), capture_output=True,
                                           text=True, timeout=8)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertNotIn(host.token, completed.stdout + completed.stderr)
                self.assertNotIn(host.cookie, completed.stdout + completed.stderr)
                return json.loads(completed.stdout)

            created = cli("init", "--config", target, "--repo", "example/project", "--branch", "main",
                          "--session", host.session, "--connect-url-file", source["connect_url_file"],
                          "--port", port)
            self.assertEqual(created["status"], "initialized")
            self.assertEqual(host.requests, [])
            checked = cli("check", "--config", target)
            self.assertEqual(checked["status"], "CONFIG_READY")
            self.assertFalse(checked["model_message_sent"])
            self.assertFalse(checked["chat_delivery_tested"])
            self.assertFalse(checked["runtime_version_verified"])
            self.assertEqual([record[2]["method"] for record in host.requests], ["session/state"])
            process = subprocess.Popen(command + ["serve", "--config", str(target)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            first_line = queue.Queue()
            reader = threading.Thread(target=lambda: first_line.put(process.stdout.readline()), daemon=True)
            reader.start()
            try:
                ready_line = first_line.get(timeout=7)
                self.assertTrue(ready_line, "serve exited before its ready record")
                ready = json.loads(ready_line)
                self.assertEqual(ready["status"], "listening")
                self.assertEqual(ready["backend"], "dsh")
                self.assertEqual(ready["port"], port)
                self.assertEqual([record[2]["method"] for record in host.requests],
                                 ["session/state", "session/state"])
                ping = cli("test", "--config", target)
                self.assertEqual(ping["status"], "PONG")
                self.assertFalse(ping["model_message_sent"])
                self.assertEqual(len(host.requests), 2)
                self.assertEqual(cli("status", "--config", target)["accepted"], 0)
                created_config = notify.load_config(target)
                generated_secret = Path(created_config["secret_file"]).read_bytes()
                self.assertEqual(self.send_push(SimpleNamespace(server_address=("127.0.0.1", port)),
                                                secret=generated_secret),
                                 (202, "queued"))
                delivered = cli("status", "--config", target)
                self.assertEqual(delivered["accepted"], 1)
                self.assertEqual(delivered["last_delivery"]["status"], "queued")
                self.assertEqual(len(self.prompt_requests(host)), 1)
                self.assertEqual(host.admissions, 1)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    remaining_stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    remaining_stdout, stderr = process.communicate(timeout=5)
                reader.join(1)
                for secret in (host.token, host.cookie):
                    self.assertNotIn(secret, remaining_stdout + stderr)


if __name__ == "__main__":
    unittest.main()
