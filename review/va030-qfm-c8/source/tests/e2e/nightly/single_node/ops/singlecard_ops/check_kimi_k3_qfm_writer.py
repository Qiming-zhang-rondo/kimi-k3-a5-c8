# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python; no vLLM initialization or model weights needed."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

MANAGER_BLOCK_SIZES = (128, 256, 384, 512, 1024)
NUM_MANAGER_PAGES = 3
GUARD_BYTES = 256
PAGE_GAP_PER_KERNEL = 256
SENTINEL = 0x5A


def _writer_slots(manager_block_size, kernel_block_size):
    """Cover kernel-page and manager-page boundaries without duplicate slots."""
    total_tokens = NUM_MANAGER_PAGES * manager_block_size
    candidates = (
        0,
        kernel_block_size - 1,
        kernel_block_size,
        manager_block_size - 1,
        manager_block_size,
        2 * manager_block_size - 1,
        2 * manager_block_size,
        total_tokens - 1,
    )
    valid = list(dict.fromkeys(candidates))
    assert all(0 <= slot < total_tokens for slot in valid)
    return valid + [-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ.pop("ASCEND_DEVICE_ID", None)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    root = Path(__file__).resolve().parents[6]
    path = root / "vllm_ascend/attention/kimi_k3_qfm.py"
    spec = importlib.util.spec_from_file_location("qfm_writer_under_test", path)
    qfm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = qfm
    spec.loader.exec_module(qfm)
    if qfm.QFM_KV_DESCALE != 1.0:
        raise AssertionError("The pinned K3 QFM writer probe requires unit KV descale")
    torch.manual_seed(417)
    for manager_block_size in MANAGER_BLOCK_SIZES:
        if manager_block_size % qfm.QFM_BLOCK:
            raise AssertionError("Probe manager block must contain whole QFM kernel blocks")
        kernel_blocks_per_manager = manager_block_size // qfm.QFM_BLOCK
        for page_gap in (0, kernel_blocks_per_manager * PAGE_GAP_PER_KERNEL):
            if page_gap % (kernel_blocks_per_manager * qfm.QFM_TILE):
                raise AssertionError("Physical page gap must split into aligned kernel-page gaps")
            page_bytes = manager_block_size * qfm.QFM_WIDTH + page_gap
            kernel_page_bytes = page_bytes // kernel_blocks_per_manager
            offset = GUARD_BYTES
            raw_bytes = NUM_MANAGER_PAGES * page_bytes
            expected = torch.full(
                (offset + raw_bytes + GUARD_BYTES,),
                SENTINEL,
                dtype=torch.uint8,
            )
            backing = expected.to("npu")
            cache = qfm.cache_view_for_manager(
                backing[offset : offset + raw_bytes],
                manager_block_size,
                page_bytes,
                kernel_block_size=qfm.QFM_BLOCK,
            )
            assert cache.shape == (
                NUM_MANAGER_PAGES * kernel_blocks_per_manager,
                1,
                qfm.QFM_WIDTH // qfm.QFM_TILE,
                qfm.QFM_BLOCK,
                qfm.QFM_TILE,
            )
            assert cache.stride(0) == kernel_page_bytes

            eager_ids = _writer_slots(manager_block_size, qfm.QFM_BLOCK)
            rows_cpu = torch.randn(len(eager_ids), qfm.QFM_WIDTH).bfloat16()
            rows = rows_cpu.to("npu")
            slots = torch.tensor(eager_ids, dtype=torch.int64, device="npu")
            scale = torch.tensor([qfm.QFM_KV_DESCALE], dtype=torch.float32, device="npu")
            quant = (
                (rows_cpu.float() / qfm.QFM_KV_DESCALE)
                .clamp(-qfm.QFM_FP8_MAX, qfm.QFM_FP8_MAX)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )

            def update_expected(
                ids,
                expected=expected,
                quant=quant,
                offset=offset,
                page_bytes=page_bytes,
                kernel_page_bytes=kernel_page_bytes,
                kernel_blocks_per_manager=kernel_blocks_per_manager,
            ):
                for row, slot in enumerate(ids):
                    if slot < 0:
                        continue
                    kernel_page, token = divmod(slot, qfm.QFM_BLOCK)
                    manager_page, kernel_page_in_manager = divmod(kernel_page, kernel_blocks_per_manager)
                    assert manager_page < NUM_MANAGER_PAGES
                    kernel_page_start = offset + manager_page * page_bytes + kernel_page_in_manager * kernel_page_bytes
                    for tile in range(qfm.QFM_WIDTH // qfm.QFM_TILE):
                        begin = kernel_page_start + tile * qfm.QFM_BLOCK * qfm.QFM_TILE + token * qfm.QFM_TILE
                        expected[begin : begin + qfm.QFM_TILE] = quant[
                            row,
                            tile * qfm.QFM_TILE : (tile + 1) * qfm.QFM_TILE,
                        ]

            qfm.write_cache(rows, slots, cache, scale)
            update_expected(eager_ids)
            torch.npu.synchronize()
            assert torch.equal(backing.cpu(), expected), "Eager manager-page writer bytes/guards mismatch"

            # Warm the native op on a side stream before graph capture.
            side = torch.npu.Stream()
            side.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side):
                qfm.write_cache(rows, slots, cache, scale)
            torch.npu.current_stream().wait_stream(side)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                qfm.write_cache(rows, slots, cache, scale)

            total_tokens = NUM_MANAGER_PAGES * manager_block_size
            replay_ids = [(slot + manager_block_size // 2 + 1) % total_tokens for slot in eager_ids[:-1]] + [-1]
            slots.fill_(-1)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(backing.cpu(), expected), "Invalid replay unexpectedly changed QFM cache bytes"

            replay_rows_cpu = torch.randn_like(rows_cpu)
            replay_quant = (
                (replay_rows_cpu.float() / qfm.QFM_KV_DESCALE)
                .clamp(-qfm.QFM_FP8_MAX, qfm.QFM_FP8_MAX)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            rows.copy_(replay_rows_cpu)
            slots.copy_(torch.tensor(replay_ids, dtype=torch.int64, device="npu"))
            graph.replay()
            update_expected(replay_ids, quant=replay_quant)
            torch.npu.synchronize()
            assert torch.equal(backing.cpu(), expected), "Replay manager-page writer bytes/guards mismatch"
            print(
                json.dumps(
                    {
                        "event": "qfm_framework_writer",
                        "pass": True,
                        "manager_block_size": manager_block_size,
                        "kernel_blocks_per_manager": kernel_blocks_per_manager,
                        "gap": page_gap,
                        "kernel_page_bytes": kernel_page_bytes,
                        "offset": offset,
                    }
                )
            )
    print(
        json.dumps(
            {
                "event": "summary",
                "pass": True,
                "scope": "native framework tile writer eager/replay only, not full model",
            }
        )
    )


if __name__ == "__main__":
    main()
