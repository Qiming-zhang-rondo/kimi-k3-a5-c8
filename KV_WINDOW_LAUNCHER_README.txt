KV window launcher environment update
====================================

This update keeps the validated v4 numerical payload pinned to
d2fcd69362788775b59a61b011cb8de712760d41 with its existing SHA256 checks.
Only the launch environment is adjusted; it does not change attention inputs,
compact grouping, mathematical references or performance measurements.
Existing successful v4 results do not need rerunning for this launcher change.

The calling shell must already have the working Python and vendor environment,
including vendor set_env.bash and the appropriate cloud_ops_turbo PYTHONPATH.
The launcher does not source vendor scripts or select a different operator.

Child-process environment handling:
  - Use ninja already available on PATH.
  - If ninja is missing, use /tmp/cloudops/bin/ninja when executable.
  - Preserve an existing TORCH_EXTENSIONS_DIR.
  - Otherwise use /tmp/cloudops/torch_ext when /tmp/cloudops exists.
  - For other deployments without that root, use an isolated cache under the
    new run directory on the filesystem selected by --work-dir.
  - Create the selected cache directory without deleting existing contents.
  - Inherit proxy variables without setting an address or guessing a port.

Optional overrides:
  --cloudops-root /path/to/cloudops
  --torch-extensions-dir /path/to/writable/jit-cache

The explicit cache argument takes precedence over TORCH_EXTENSIONS_DIR.
Resolved ninja/cache choices are printed and recorded in launcher.json under
test_environment. Proxy URLs and the full environment are not logged there.
These changes apply only to the test subprocess, not the parent shell.

Keep the proxy that works from the specific machine. Downloads happen in the
launcher process using its inherited proxy configuration. No packages, CANN,
driver, model code or kernel binaries are installed or upgraded by this script.
