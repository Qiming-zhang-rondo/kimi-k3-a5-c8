"""Verify the unmodified, previously tested BF16/C4 experiment before importing it."""
import hashlib
from pathlib import Path
import sys

PAYLOAD_COMMIT = '4a5fd67b9a78766bebfa564782ea2193bc22af1a'

FILES = {
    "test_kv_hot_fused_a5.py": "a64183fa77fbdaa8085c2ff03b90723de6e9f31deffdfb2b91fb9a42ce22a2e2",
    "kv_window_backend.py": "20c33c19612da73484ea7c57e05b2a78eba68e7bc886db6daf0997976a530bc4",
    "kv_window_common.py": "7357ca29ee22f71445b1d9a3d5eddf8c2d6c0a0dab134b0e2e4d42c36e3dd724",
    "kv_hot_backend.py": "90a9a005458ebb2c8b837f103641b3897a8c99d29fa4f22ae4254dc6d9cb61fa",
    "kv_c4_scale_backend.py": "5002af47b6f7394f84f6b5d2d65eeeaf4de8097df279f9cf0328b252f87149a5",
    "kv_native_twopass.py": "e0c05f004473e4e10f20fe0bc6616b6c381519361f47ae31ec38d7cc5f05498e",
    "kv_window_capture.py": "b872f9a378a22a27c3ff970426c33718d3c20958724ee018d383b4bb51216bb7",
    "kv_native_store.py": "94e33ae49f72bfcf5efc0c1914e5ca9f312e6afce1ed9442b22decea56ef85d2",
    "kv_native_merge.py": "55c5bad52891dd539e9e5a1acde942aca7decf4019e20d658a08b12b7ad4468a",
    "NATIVE_TWOPASS_README.txt": "5d2cb94017d8193add0938024b6529c684ed1bde491f4db246b940fc31d12283",
    "inspect_kv_fusion_capability.py": "d0f7bb3052a29380b363e4d177a505299c4d5a91f3d607e2e2f66872a462e6f9",
    "kv_native_hot_merge.py": "20a20e022bfd70b590136764cf227283b7ed106d454fb838ad2653c559685494",
    "kv_native_partition.py": "1b83122344bd1fd2ff07544f9fcf8678e0edd6e4f37e583a71940a32b363d7db",
    "test_kv_native_twopass_a5.py": "fd54ff5e145dedce5fbc35d53b7143a26760a7aced3a1ee269ae91dab70b73c5",
    "kv_native_hot_prepare.py": "f5539394b4885082e4b3cc0e87832719c4c53bd37768d97dda8e8814bb55bb24",
    "kv_bf16_hot_ab.py": "c4cbc22411dc17f42b0adfa77776edd4644b1adc802b5164b69b84edb66566e7",
    "test_kv_bf16_hot_ab_a5.py": "a19df2c4fd40faa92964b2e264d3d64d1928935f2124357ead505d83af285731",
    "BF16_HOT_AB_README.txt": "ee0d0a63730c905274b8a6449ae901f5acc1e0f10290d944260776b23bf9b47f"
}


def verify_payload(directory, manifest=None):
    root = Path(directory).expanduser().resolve()
    expected = FILES if manifest is None else manifest
    records = {}
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing pinned payload file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f"Payload SHA256 mismatch: {name}; expected commit {PAYLOAD_COMMIT}")
        records[name] = actual
    return records


def activate_payload(directory):
    root = Path(directory).expanduser().resolve()
    records = verify_payload(root)
    for name in FILES:
        if name.endswith('.py') and name[:-3] in sys.modules:
            loaded = Path(getattr(sys.modules[name[:-3]], '__file__', '')).resolve()
            if loaded != root / name:
                raise RuntimeError(f"A different payload module was already imported: {name}")
    sys.path.insert(0, str(root))
    return dict(commit=PAYLOAD_COMMIT, directory=str(root), sha256=records)
