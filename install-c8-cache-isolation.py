"""Apply the C8 hybrid cache/DSpark length repair to the 2026-10-29 test bundle.

Self-contained; no downloads, model loading, tests, or service restart.
Refuses unknown source changes. Backups are made before applying.
NPU and whole-model validation are still required.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

STATES = {
    "vllm_ascend/patch/platform/patch_kv_cache_utils.py": {
        "before": "8ea043d47051ff3bee1a224cce2b092dfdb85bf1",
        "after": "2bf2a2635e02571f7eca0ff7da212799782daa1c",
    },
    "vllm_ascend/worker/v2/attn_utils.py": {
        "before": "877a06d23a3fd8e0da70d401cb2c9a2fe4d21d56",
        "after": "4f3e1ce19387319ddd7e4c04736b4c06ba538b31",
    },
    "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py": {
        "before": "d26c2f9b50e7b67ce929ff1b5785fa9951f313dd",
        "after": "a246e90a4b6baf17489b1744a20b845d4077ef7f",
    },
    "vllm_ascend/core/kimi_k3_cache_layout.py": {
        "before": None,
        "after": "56fe9e531bf05e79444e5ec078a79038005967c5",
    },
}
DEPENDENCIES = {
    "vllm_ascend/attention/mla_v1.py": "b3411936689e64923e2c22b99af9442322d8bb50",
    "vllm_ascend/attention/kimi_k3_sfa_c8.py": "15478d8e8cecfb7afae1b443c202b1f68d222467",
    "vllm_ascend/attention/kimi_k3_sfa_c8_graph.py": "f03235fde821473b6c4df5b3ce748f4358927658",
}
PATCH = 'diff --git a/vllm_ascend/patch/platform/patch_kv_cache_utils.py b/vllm_ascend/patch/platform/patch_kv_cache_utils.py\nindex 8ea043d47..2bf2a2635 100644\n--- a/vllm_ascend/patch/platform/patch_kv_cache_utils.py\n+++ b/vllm_ascend/patch/platform/patch_kv_cache_utils.py\n@@ -23,6 +23,7 @@ from vllm.v1.kv_cache_interface import (\n     get_kv_cache_spec_kind,\n )\n \n+from vllm_ascend.core.kimi_k3_cache_layout import get_kimi_k3_c8_cache_layout\n from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable\n from vllm_ascend.models.deepseek_v41.cache_config import (\n     get_deepseek_v41_kv_cache_config,\n@@ -536,6 +537,9 @@ def _ascend_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int\n     layout, so using the upstream value changes ``num_blocks`` during the\n     re-plan and leaves ranks inconsistent.\n     """\n+    c8_layout = get_kimi_k3_c8_cache_layout(kv_cache_groups)\n+    if c8_layout is not None:\n+        return c8_layout[0]\n     if is_deepseek_v41_cache(kv_cache_groups):\n         return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)\n     if _get_glm5_next_cache_layout(kv_cache_groups) is not None:\n@@ -597,7 +601,46 @@ def _ascend_get_kv_cache_config_from_groups(\n     ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config`` and is\n     dropped by worker pickle IPC, which never reads it.\n     """\n-    if is_deepseek_v41_cache(kv_cache_groups):\n+    c8_layout = get_kimi_k3_c8_cache_layout(kv_cache_groups)\n+    if c8_layout is not None:\n+        bytes_per_block, layer_offsets = c8_layout\n+        num_blocks = may_override_num_blocks(vllm_config, available_memory // bytes_per_block)\n+        size = num_blocks * bytes_per_block\n+        if size > available_memory:\n+            raise ValueError(\n+                f"C8 hybrid cache needs {size} bytes for {num_blocks} blocks, "\n+                f"but the KV cache budget is {available_memory} bytes."\n+            )\n+        tensors = []\n+        for group in kv_cache_groups:\n+            for name in group.layer_names:\n+                spec = (\n+                    group.kv_cache_spec.kv_cache_specs[name]\n+                    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n+                    else group.kv_cache_spec\n+                )\n+                tensors.append(\n+                    KVCacheTensor(\n+                        size=size,\n+                        layers=[name],\n+                        offset=num_blocks * layer_offsets[name],\n+                        layer_stride=num_blocks * spec.page_size_bytes,\n+                        block_stride=spec.page_size_bytes,\n+                    )\n+                )\n+        kv_cache_config = KVCacheConfig(\n+            num_blocks=num_blocks,\n+            kv_cache_tensors=tensors,\n+            kv_cache_groups=kv_cache_groups,\n+            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,\n+        )\n+        logger.info(\n+            "C8 hybrid cache: isolated format regions, %d bytes per global block, %d blocks, %d bytes allocated.",\n+            bytes_per_block,\n+            num_blocks,\n+            size,\n+        )\n+    elif is_deepseek_v41_cache(kv_cache_groups):\n         kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)\n     elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:\n         kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)\ndiff --git a/vllm_ascend/worker/v2/attn_utils.py b/vllm_ascend/worker/v2/attn_utils.py\nindex 877a06d23..4f3e1ce19 100644\n--- a/vllm_ascend/worker/v2/attn_utils.py\n+++ b/vllm_ascend/worker/v2/attn_utils.py\n@@ -52,6 +52,7 @@ from vllm_ascend.attention.utils import (\n     AscendCommonAttentionMetadata,\n     get_sfa_qsfa_packed_head_dim,\n )\n+from vllm_ascend.core.kimi_k3_cache_layout import get_kimi_k3_c8_cache_layout\n from vllm_ascend.core.kv_cache_interface import (\n     AscendIndexerKPoolTailSpec,\n     AscendMLAAttentionSpec,\n@@ -649,6 +650,38 @@ def _allocate_kv_cache(\n     has_mamba = any(isinstance(spec, MambaSpec) for spec in layer_kv_cache_spec.values())\n     has_attention = any(isinstance(spec, AttentionSpec) for spec in layer_kv_cache_spec.values())\n     use_hybrid_layout = has_mamba and has_attention\n+    c8_layout = get_kimi_k3_c8_cache_layout(kv_cache_config.kv_cache_groups)\n+    if c8_layout is not None:\n+        # Fail before allocating if an old planner still overlays incompatible\n+        # formats. Merely creating a correct C8 view cannot repair that alias.\n+        bytes_per_block, layer_offsets = c8_layout\n+        expected_size = kv_cache_config.num_blocks * bytes_per_block\n+        seen_layers: set[str] = set()\n+        region_dims: dict[int, tuple[int, int]] = {}\n+        for descriptor in kv_cache_config.kv_cache_tensors:\n+            for index, name in enumerate(get_kv_cache_tensor_layers(descriptor)):\n+                spec = layer_kv_cache_spec[name]\n+                start = descriptor.offset + index * descriptor.layer_stride\n+                if (\n+                    name in seen_layers\n+                    or descriptor.size != expected_size\n+                    or start != kv_cache_config.num_blocks * layer_offsets[name]\n+                    or descriptor.block_stride != spec.page_size_bytes\n+                    or descriptor.layer_stride != kv_cache_config.num_blocks * spec.page_size_bytes\n+                ):\n+                    raise ValueError(\n+                        f"C8 hybrid cache region for {name} does not match the isolated-format plan; "\n+                        "update the cache planner and worker together and restart."\n+                    )\n+                seen_layers.add(name)\n+                if isinstance(spec, MLAAttentionSpec) and not getattr(spec, "cache_sparse_sfa_c8", False):\n+                    # The MLA spec stores only total width. Equal specs may\n+                    # share a region only if their actual K/V split agrees.\n+                    dims = _get_attention_kv_cache_dims(name, spec)\n+                    if region_dims.setdefault(start, dims) != dims:\n+                        raise ValueError(f"MLA layers sharing C8 hybrid region {start} have different K/V splits.")\n+        if seen_layers != set(layer_offsets):\n+            raise ValueError("C8 hybrid cache descriptors do not cover every planned layer.")\n \n     # The restored DeepSeek-V4 planner on main computes capacity for one\n     # shared-tuple backing and emits every KVCacheTensor as a view into it.\n@@ -748,7 +781,7 @@ def _allocate_kv_cache(\n         # model\'s Attention/Mamba caches. Keep HiddenStateCacheSpec off the\n         # #51718 hybrid backing so float32 SSM writes cannot overlay bfloat16\n         # hidden states, and size each dump from its own page.\n-        if any(is_hidden_state_cache_spec(layer_kv_cache_spec[ln]) for ln in shared_names):\n+        if c8_layout is None and any(is_hidden_state_cache_spec(layer_kv_cache_spec[ln]) for ln in shared_names):\n             for layer_idx, layer_name in enumerate(shared_names):\n                 layer_spec = layer_kv_cache_spec[layer_name]\n                 if is_hidden_state_cache_spec(layer_spec) or hybrid_backing is None:\n@@ -1250,7 +1283,7 @@ def build_attn_metadata_wrapper():\n \n \n @contextmanager\n-def build_draft_attn_metadata_factory(positions, pad, is_prefilling, *, attn_state=None):\n+def build_draft_attn_metadata_factory(positions, pad, is_prefilling, *, attn_state=None, exact_seq_lens=False):\n     """Wrap build_attn_metadata with Ascend draft-model context.\n \n     The generic (Ascend) ``build_attn_metadata`` reads ``positions`` inside the\n@@ -1262,6 +1295,19 @@ def build_draft_attn_metadata_factory(positions, pad, is_prefilling, *, attn_sta\n     raw = _BUILD_ATTN_METADATA_MODULE.build_attn_metadata  # cache\n \n     def build_attn_metadata(*args, **kwargs):\n+        if exact_seq_lens:\n+            # BF16 MLA/FIA consumes a host length list, unlike the C8 reader\n+            # which uses device lengths and a host upper bound only for sizing.\n+            # Snapshot after draft rejection/rewind, BEFORE the builder derives\n+            # decode/CP metadata. This intentional small D2H synchronization\n+            # stays outside captured forward/replay; do not replace with an\n+            # upper bound. Nested draft factories reuse the exact CPU mirror.\n+            if kwargs.get("seq_lens_np") is None:\n+                lengths = kwargs["seq_lens"][: kwargs["num_reqs"]].to(device="cpu", copy=True)\n+                kwargs["seq_lens_np"] = lengths.numpy()\n+            else:\n+                lengths = torch.from_numpy(kwargs["seq_lens_np"][: kwargs["num_reqs"]])\n+            kwargs["seq_lens_cpu_upper_bound"] = lengths\n         kwargs["positions"] = positions[:pad]\n         kwargs["is_prefilling"] = is_prefilling\n         kwargs["attn_state"] = attn_state\ndiff --git a/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py b/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py\nindex d26c2f9b5..a246e90a4 100644\n--- a/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py\n+++ b/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py\n@@ -124,6 +124,7 @@ class AscendDSparkSpeculator(DSparkSpeculator):\n                 num_tokens_padded,\n                 is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool),\n                 attn_state=AscendAttentionState.ChunkedPrefill,\n+                exact_seq_lens=self.attn_architecture == "MLA",\n             ),\n         ):\n             attn_metadata = super()._build_draft_attn_metadata(\n@@ -158,6 +159,7 @@ class AscendDSparkSpeculator(DSparkSpeculator):\n                 num_tokens_padded,\n                 is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool),\n                 attn_state=AscendAttentionState.ChunkedPrefill,\n+                exact_seq_lens=self.attn_architecture == "MLA",\n             ),\n         ):\n             attn_metadata = super()._build_draft_attn_metadata(num_reqs_padded=num_reqs_padded, **kwargs)\n@@ -216,7 +218,10 @@ class AscendDSparkSpeculator(DSparkSpeculator):\n         with (\n             build_attn_metadata_wrapper(),\n             build_draft_attn_metadata_factory(\n-                self.input_buffers.positions, self.max_num_tokens, torch.from_numpy(self.input_batch.is_prefilling_np)\n+                self.input_buffers.positions,\n+                self.max_num_tokens,\n+                torch.from_numpy(self.input_batch.is_prefilling_np),\n+                exact_seq_lens=self.attn_architecture == "MLA",\n             ),\n         ):\n             return super().propose(\ndiff --git a/vllm_ascend/core/kimi_k3_cache_layout.py b/vllm_ascend/core/kimi_k3_cache_layout.py\nnew file mode 100644\nindex 000000000..56fe9e531\n--- /dev/null\n+++ b/vllm_ascend/core/kimi_k3_cache_layout.py\n@@ -0,0 +1,61 @@\n+# SPDX-License-Identifier: Apache-2.0\n+# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project\n+\n+from collections import Counter\n+\n+from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, MambaSpec, UniformTypeKVCacheSpecs\n+\n+\n+def get_kimi_k3_c8_cache_layout(\n+    groups: list[KVCacheGroupSpec],\n+) -> tuple[int, dict[str, int]] | None:\n+    """Budget disjoint storage regions for incompatible hybrid cache formats.\n+\n+    Packed C8 uses page-major storage; ordinary MLA uses whole-pool K/V\n+    planes and Mamba uses whole-pool state planes. Different block IDs do\n+    not isolate these layouts when they alias the same byte range. Keep\n+    the existing native tensor layouts, but allow storage reuse only among\n+    identical specs in different scheduler groups.\n+\n+    Return the total bytes per global block and each layer\'s region offset\n+    *per block*. Multiply both by num_blocks when building descriptors.\n+    Logical groups/block tables are unchanged. Non-C8 models are untouched.\n+    """\n+    layer_groups = [\n+        [\n+            (\n+                name,\n+                group.kv_cache_spec.kv_cache_specs[name]\n+                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n+                else group.kv_cache_spec,\n+            )\n+            for name in group.layer_names\n+        ]\n+        for group in groups\n+    ]\n+    specs = [spec for layers in layer_groups for _, spec in layers]\n+    if not any(getattr(spec, "cache_sparse_sfa_c8", False) for spec in specs) or not any(\n+        isinstance(spec, MambaSpec) for spec in specs\n+    ):\n+        return None\n+\n+    pool_widths: dict[KVCacheSpec, int] = {}\n+    for layers in layer_groups:\n+        for spec, count in Counter(spec for _, spec in layers).items():\n+            pool_widths[spec] = max(pool_widths.get(spec, 0), count)\n+\n+    pool_offsets: dict[KVCacheSpec, int] = {}\n+    bytes_per_block = 0\n+    for spec, width in pool_widths.items():\n+        pool_offsets[spec] = bytes_per_block\n+        bytes_per_block += width * spec.page_size_bytes\n+\n+    layer_offsets: dict[str, int] = {}\n+    for layers in layer_groups:\n+        indices: Counter[KVCacheSpec] = Counter()\n+        for name, spec in layers:\n+            if name in layer_offsets:\n+                raise ValueError(f"Duplicate hybrid cache layer: {name}.")\n+            layer_offsets[name] = pool_offsets[spec] + indices[spec] * spec.page_size_bytes\n+            indices[spec] += 1\n+    return bytes_per_block, layer_offsets\n'


def blob(path):
    if not path.exists():
        return None
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def run(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--va-source", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Read-only applicability check"
    )
    args = parser.parse_args()
    if args.va_source is None:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or spec.origin is None:
            parser.error("Cannot locate editable VA; provide --va-source")
        root = Path(spec.origin).resolve().parent.parent
    else:
        root = args.va_source.resolve()
    if not (root / "vllm_ascend").is_dir():
        parser.error(f"Not a VA source directory: {root}")

    for name in {*STATES, *DEPENDENCIES}:
        path = root / name
        if any(
            parent.is_symlink() for parent in [path, *path.parents] if parent != root
        ):
            raise RuntimeError(f"Refusing symlink in affected path: {name}")
    bad_deps = [
        name for name, expected in DEPENDENCIES.items() if blob(root / name) != expected
    ]
    if bad_deps:
        raise RuntimeError(
            "Requires the 2026-10-29 C8/DSpark test bundle; differing reader sources: "
            + ", ".join(bad_deps)
        )
    actual = {name: blob(root / name) for name in STATES}
    if all(actual[name] == state["after"] for name, state in STATES.items()):
        print("Already applied. No changes.")
        return 0
    mismatches = [
        name for name, state in STATES.items() if actual[name] != state["before"]
    ]
    if mismatches:
        raise RuntimeError(
            "Unknown or partial edits; no changes made: " + ", ".join(mismatches)
        )
    in_repo = run(root, "rev-parse", "--show-toplevel")
    if in_repo.returncode == 0:
        staged = run(root, "diff", "--cached", "--name-only", "--", *STATES)
        if staged.returncode or staged.stdout.strip():
            raise RuntimeError("Affected files have staged changes; no changes made.")
    with tempfile.TemporaryDirectory(prefix="k3-c8-cache-check-") as folder:
        patch_file = Path(folder) / "fix.patch"
        patch_file.write_text(PATCH)
        check = run(root, "apply", "--check", str(patch_file))
        if check.returncode:
            raise RuntimeError(check.stderr.strip())
        if args.check:
            print(f"Applicable: {root}. Four runtime files; no changes made.")
            return 0
        backup = Path(tempfile.mkdtemp(prefix="k3-c8-cache-backup-"))
        for name, state in STATES.items():
            if state["before"] is not None:
                target = backup / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, target)
        (backup / "manifest.json").write_text(json.dumps(STATES, indent=2))
        print(f"Source: {root}\nBackup: {backup}", flush=True)
        applied = run(root, "apply", str(patch_file))
        if applied.returncode:
            raise RuntimeError(
                f"Patch failed; backup retained at {backup}: {applied.stderr.strip()}"
            )
        if any(blob(root / name) != state["after"] for name, state in STATES.items()):
            raise RuntimeError(
                f"Post-apply hash mismatch; inspect backup {backup} before proceeding."
            )
    print("Applied: isolated cache regions + exact DSpark MLA lengths.")
    print("Cache block capacity can decrease within the SAME memory budget.")
    print(
        "Exact draft lengths add a small host synchronization; measure its cost on A5."
    )
    print(
        "Restart workers before model testing. This installer ran NO NPU/model tests."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"Installation refused: {exc}")
