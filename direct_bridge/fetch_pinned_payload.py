#!/usr/bin/env python3
"""Fetch only the eighteen SHA256-pinned old BF16 harness files; no NPU work."""
import argparse
import hashlib
from pathlib import Path
import shutil
import ssl
import tempfile
from urllib.request import urlopen

from pinned_payload import FILES, PAYLOAD_COMMIT, verify_payload

BASE = ("https://raw.githubusercontent.com/Qiming-zhang-rondo/kimi-k3-a5-c8/"
        + PAYLOAD_COMMIT + "/kv-native-twopass/hotkv/")
MAX_FILE = 2 * 1024 * 1024


def fetch(output, ca_file=None):
    output = Path(output).expanduser().resolve()
    if output.exists():
        verify_payload(output)
        return output  # Reuse only an already complete, byte-identical payload.
    context = ssl.create_default_context(cafile=ca_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sfa-pinned-download-", dir=output.parent))
    try:
        for name, expected in FILES.items():
            with urlopen(BASE + name, timeout=30, context=context) as response:
                data = response.read(MAX_FILE + 1)
            if len(data) > MAX_FILE or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Pinned payload download failed size/SHA256 validation: " + name)
            (staging / name).write_bytes(data)
        verify_payload(staging)
        if output.exists():
            raise ValueError("Destination appeared during download; refusing to replace it")
        staging.rename(output)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only the temporary directory created above.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ca-file", type=Path, help="Optional trusted CA bundle; TLS verification stays enabled")
    args = parser.parse_args(argv)
    path = fetch(args.output, args.ca_file)
    print("PINNED_PAYLOAD_VERIFIED:", path, "commit=" + PAYLOAD_COMMIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
