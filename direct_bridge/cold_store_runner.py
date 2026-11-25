"""Opt-in writer candidate; preserve native quantization and the old byte oracle."""
import numpy as np


PART_NAMES=("old","recent_global","recent_ring","old_present","recent_present","invalid_rows")


def _cpu(value):
    return value.detach().cpu().numpy().copy()


def exact_field(actual, expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    same_shape=actual.shape==expected.shape
    same_dtype=actual.dtype==expected.dtype
    bad=np.argwhere(actual!=expected) if same_shape else np.empty((0,actual.ndim),np.int64)
    return dict(passed=bool(same_shape and same_dtype and not len(bad)),
        expected_shape=list(expected.shape),actual_shape=list(actual.shape),
        expected_dtype=str(expected.dtype),actual_dtype=str(actual.dtype),
        mismatch_count=int(len(bad)) if same_shape else int(max(actual.size,expected.size)),
        first_bad_positions=bad[:16].tolist())


def partition_reference(runner):
    from kv_native_twopass import partition_numpy
    values=partition_numpy(runner.indices_cpu,runner.positions_cpu,runner.window,
                           runner.capacity,runner.index_capacity)
    values["invalid_rows"]=np.zeros(len(runner.indices_cpu),np.bool_)
    return {name:values[name] for name in PART_NAMES}


def cache_views(runner):
    byte=runner.base.torch.uint8
    return {
        "cold":runner.cold_cache.view(byte).view(-1,runner.cold_cache.shape[-1]),
        "hot_latent":runner.hot_cache.view(byte).view(-1,1024),
        "hot_rope":runner.hot_rope.view(byte).view(-1,128),
    }


def _poison_parts(outputs, expected):
    for name,value in outputs.items():
        # These controlled all-hot cases have uniform presence in each row.
        value.fill_(not bool(expected[name].flat[0]) if expected[name].dtype==np.bool_ else -17)


def make_cold_store_runner_class(direct_runner_class, prepare_class=None):
    class ColdStoreRunner(direct_runner_class):
        def __init__(self,*args,**kwargs):
            if kwargs.get("writer","hot_store_partition")!="hot_store_partition":
                raise ValueError("Cold scatter fusion requires the shared hot_store_partition producer")
            super().__init__(*args,**kwargs)
            cls=prepare_class
            if cls is None:
                from cold_store_prepare import PreparedColdBf16HotStorePartition
                cls=PreparedColdBf16HotStorePartition
            packed_template=self.variant._quantize(self.base.current_kv,self.base.current_rope)[0]
            self._combined_prepare=cls(self._hot_prepare,self.cold_cache,self.base.current_slots,packed_template)
            if self._combined_prepare.outputs is not self._hot_prepare.outputs:
                raise ValueError("New producer must write the exact partition dictionary consumed by run")
            self.metadata.update(writer="cold_scatter_hot_prepare",native_c4_writer_unchanged=False,
                native_c4_quantizer_unchanged=True,native_c4_packed_format_unchanged=True,
                native_quantize_calls_per_step=1,cold_index_copy_api_calls_per_step=0,
                prepare_kernel_launches_per_step=1,
                write_scope="unchanged native quantize+pack, then one cold scatter/BF16 hot store/partition call",
                writer_implementation=dict(self._combined_prepare.metadata),
                writer_validation="PENDING_A5_BYTE_AND_GRAPH_GATES")

        def _write(self):
            if not hasattr(self,"_combined_prepare"):
                return super()._write()
            packed=self.variant._quantize(self.base.current_kv,self.base.current_rope)[0]
            self._combined_prepare.call(packed)
            return self.cold_cache.view(self.base.torch.uint8)

        def partition_diagnostics(self):
            # Calling the inherited method would exercise old_prepare.call(),
            # not this writer. Poison all six shared outputs and use _write().
            views=cache_views(self)
            saved={name:value.clone() for name,value in views.items()}
            outputs=self._hot_prepare.outputs
            saved_parts={name:value.clone() for name,value in outputs.items()}
            expected=partition_reference(self)
            fault=False
            try:
                _poison_parts(outputs,expected)
                self._write()
                fields={name:exact_field(_cpu(outputs[name]),expected[name]) for name in PART_NAMES}
                return dict(passed=all(item["passed"] for item in fields.values()),fields=fields,
                    producer="cold_scatter_bf16_hot_store_partition_after_output_poison",
                    invalid_rows_checked=True,scope="all integer/bool outputs versus unchanged CPU partition oracle")
            except Exception as exc:
                fault=any(code in str(exc) for code in ("507015","ACL_ERROR_RT_AICORE_EXCEPTION","GM invalid"))
                raise
            finally:
                if not fault:
                    for name,value in views.items(): value.copy_(saved[name])
                    for name,value in outputs.items(): value.copy_(saved_parts[name])

    return ColdStoreRunner


def writer_graph_control(runner,pipeline):
    """Prove the new cold scatter is captured even when attention reads hot tokens.

    Poison only current write slots. One decode/chunk step cannot regenerate
    all historical KV, so poisoning entire history would test a false contract.
    The existing full-cache eager byte oracle remains an independent gate.
    """
    torch=runner.base.torch
    pipeline()
    torch.npu.synchronize()
    views=cache_views(runner)
    expected_cache={name:_cpu(value) for name,value in views.items()}
    expected_parts=partition_reference(runner)
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_output=pipeline()
    torch.npu.synchronize()
    for name,value in views.items():
        slots=runner.base.current_slots if name=="cold" else runner.hot_slots
        value.index_fill_(0,slots,165)
    _poison_parts(runner._hot_prepare.outputs,expected_parts)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    fields={name:exact_field(_cpu(value),expected_cache[name]) for name,value in views.items()}
    fields.update({"partition_"+name:exact_field(_cpu(runner._hot_prepare.outputs[name]),expected_parts[name])
                   for name in PART_NAMES})
    # Keep captured graph outputs and their graph-owned allocations alive until
    # every device comparison has completed; never replace data with a reference.
    del captured_output
    return dict(passed=all(value["passed"] for value in fields.values()),fields=fields,
        scope="one full pipeline graph; current cold/hot/RoPE slots and all partition outputs poisoned before replay",
        historical_slots_poisoned=False,standalone_cold_index_copy_in_candidate=False)
