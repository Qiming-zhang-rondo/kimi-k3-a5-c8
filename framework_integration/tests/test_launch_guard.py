"""Host-only proof that environment exports cannot silently claim recent ON."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfa_recent128.launch_guard import require_explicit_config


class LaunchGuardTests(unittest.TestCase):
    def test_absent_request_preserves_default_off_and_arbitrary_vendor_config(self):
        for config in (None,{},dict(dspark={"enabled":False})):
            require_explicit_config(config,environ={})

    def test_each_environment_only_request_raises_before_silent_off(self):
        for name in ("SFA_RECENT128_CONFIG","SFA_RECENT128_LIB_DIR"):
            env={name:"caller-only-value"}
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,"additional_config.*accuracy_eager.*PD/graph"):
                require_explicit_config({},environ=env)
            self.assertEqual(env,{name:"caller-only-value"})

    def test_explicit_off_or_on_remains_authoritative_without_env_translation(self):
        for window in (0,128):
            config=dict(sfa_recent128=dict(window=window))
            original=config["sfa_recent128"]
            require_explicit_config(config,environ={"SFA_RECENT128_CONFIG":"not even JSON"})
            self.assertIs(config["sfa_recent128"],original)
            self.assertEqual(config,{"sfa_recent128":{"window":window}})

    def test_no_environment_value_is_copied_into_error_or_configuration(self):
        private="private-configuration-value-do-not-print"
        with self.assertRaises(RuntimeError) as error:
            require_explicit_config(None,environ={"SFA_RECENT128_CONFIG":private})
        self.assertNotIn(private,str(error.exception))


if __name__=="__main__":
    unittest.main()
