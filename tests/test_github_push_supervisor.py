"""Pure CPU helper tests; no credentials, tunnel, GitHub writes, or chat calls."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("github_push_supervisor", TOOLS / "github_push_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        secret = self.root / "secret"
        secret.write_text("a-test-secret-with-at-least-thirty-two-bytes")
        self.config = dict(repo="owner/repo", thread="test-thread", secret_file=str(secret))
        self.endpoint = "https://small-safe-example.trycloudflare.com/github"

    def test_extracts_exact_quick_tunnel_host_only(self):
        self.assertEqual(supervisor.extract_endpoint("| https://small-safe-example.trycloudflare.com |\n"), self.endpoint)
        for invalid in ("https://small.trycloudflare.com.evil", "https://small.trycloudflare.com/path",
                        "http://small.trycloudflare.com", "https://small.trycloudflare.com@evil"):
            self.assertIsNone(supervisor.extract_endpoint(invalid))

    def test_credential_helper_is_noninteractive_and_secret_stays_in_memory(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, "username=test\npassword=TEST_TOKEN\n", ""))
        self.assertEqual(supervisor.credential_token(runner), "TEST_TOKEN")
        argv, kwargs = runner.call_args.args[0], runner.call_args.kwargs
        self.assertEqual(argv, ["git", "credential", "fill"])
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["env"]["GIT_ASKPASS"], "false")
        self.assertNotIn("TEST_TOKEN", str(argv))

    def test_curl_receives_auth_endpoint_and_payload_only_on_stdin(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, '{"id":123}\n201', ""))
        api = supervisor.GitHubApi("TEST_TOKEN", runner)
        self.assertEqual(api.request("POST", "/repos/owner/repo/hooks", {"value":'a"b\\c\n'}), {"id":123})
        self.assertEqual(runner.call_args.args[0], ["curl", "--disable", "--config", "-"])
        request = runner.call_args.kwargs["input"]
        self.assertIn("Authorization: Bearer TEST_TOKEN", request)
        self.assertIn("https://api.github.com/repos/owner/repo/hooks", request)
        self.assertNotIn("--insecure", request)

    def test_transport_errors_never_include_subprocess_secret_output(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 7, "TEST_TOKEN", "private endpoint"))
        with self.assertRaisesRegex(supervisor.SupervisorError, "^github_transport_failed$"):
            supervisor.GitHubApi("TEST_TOKEN", runner).request("GET", "/repos/owner/repo")

    def test_first_registration_creates_only_one_push_hook_and_private_owned_state(self):
        api = Mock()
        api.request.side_effect = [{"permissions":{"admin":True}}, {"id":321}]
        self.assertEqual(supervisor.register_hook(self.config, self.endpoint, self.root, api), 321)
        self.assertEqual([call.args[:2] for call in api.request.call_args_list],
            [("GET","/repos/owner/repo"),("POST","/repos/owner/repo/hooks")])
        payload = api.request.call_args.args[2]
        self.assertEqual(payload["events"], ["push"])
        self.assertIs(payload["active"], True)
        self.assertEqual(payload["config"]["insecure_ssl"], "0")
        self.assertEqual(os.stat(self.root / "hook_state.json").st_mode & 0o777, 0o600)
        self.assertNotIn("secret", (self.root / "hook_state.json").read_text())

    def test_restart_patches_only_previously_owned_id(self):
        supervisor.private_json(self.root / "hook_state.json",
            dict(version=1, repo="owner/repo", thread="test-thread", hook_id=321))
        api = Mock()
        api.request.side_effect = [{"id":321,"name":"web"}, {"id":321}]
        self.assertEqual(supervisor.register_hook(self.config, self.endpoint, self.root, api), 321)
        self.assertEqual([call.args[:2] for call in api.request.call_args_list],
            [("GET","/repos/owner/repo/hooks/321"),("PATCH","/repos/owner/repo/hooks/321")])

    def test_foreign_state_or_missing_admin_never_mutates_hooks(self):
        state = dict(version=1, repo="another/repo", thread="test-thread", hook_id=9)
        supervisor.private_json(self.root / "hook_state.json", state)
        api = Mock()
        with self.assertRaisesRegex(supervisor.SupervisorError, "owned_hook_state_invalid"):
            supervisor.register_hook(self.config, self.endpoint, self.root, api)
        api.request.assert_not_called()
        (self.root / "hook_state.json").unlink()
        api.request.return_value = {"permissions":{"admin":False}}
        with self.assertRaisesRegex(supervisor.SupervisorError, "repository_admin_required"):
            supervisor.register_hook(self.config, self.endpoint, self.root, api)
        self.assertEqual(api.request.call_count, 1)

    def test_tunnel_output_drives_registration_and_exit_cleans_up_without_printing_url(self):
        server = Mock(server_address=("127.0.0.1",8765))
        child = Mock(stdout=io.StringIO("| https://small-safe-example.trycloudflare.com |\n"))
        child.poll.return_value = 0
        child.wait.return_value = 0
        config = dict(self.config, branch="sfa-framework-recent128", port=8765)
        output = io.StringIO()
        with patch.object(supervisor.notify,"make_server",return_value=server), \
             patch.object(supervisor.notify,"receiver_from_config"), \
             patch.object(supervisor.threading,"Thread"), \
             patch.object(supervisor.subprocess,"Popen",return_value=child), \
             patch.object(supervisor,"credential_token",return_value="FAKE"), \
             patch.object(supervisor,"GitHubApi"), \
             patch.object(supervisor,"register_hook",return_value=321) as register, \
             patch.object(supervisor.signal,"signal"), redirect_stdout(output):
            self.assertEqual(supervisor.supervise(config,"/fake/cloudflared",self.root),1)
        register.assert_called_once()
        self.assertEqual(register.call_args.args[1],self.endpoint)
        self.assertIn('"status": "REGISTERED"',output.getvalue())
        self.assertNotIn("trycloudflare",output.getvalue())
        health=json.loads((self.root/"health.json").read_text())
        self.assertEqual(health["status"],"FAILED")
        self.assertEqual(health["error"],"tunnel_exited")
        self.assertNotIn("url",health)
        self.assertEqual(os.stat(self.root/"cloudflared.log").st_mode & 0o777,0o600)
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()
        child.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
