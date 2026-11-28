"""Install after MXLLM's existing MXFP4 worker patches.

Window zero never imports the NPU runtime or loads a candidate library. The
write tap is inside the original native RMSNorm/RoPE implementation, before
MXFP4 quantization. A packed-cache copy is never used as high precision KV.
Serving uses the explicit Prolog producer scope instead of this native tap.
"""
from functools import wraps

from .contract import parse_recent128_config
from .launch_guard import require_explicit_config


def capture_native_kv(impl, latent, rope):
    runtime = getattr(impl, "_recent128_runtime", None)
    if runtime is not None:
        if getattr(getattr(impl, "_recent128_config", None), "execution", None) == "serving":
            # The serving producer writes its own BF16 ring in PrologV3.
            # Never manufacture raw history from a native/packed-cache tap.
            return
        metadata = getattr(impl, "_recent128_write_metadata", None)
        if metadata is None:
            raise RuntimeError("recent128 raw KV tap ran outside native exec_kv")
        runtime.capture(latent, rope, metadata)


def install():
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    if getattr(AscendSFAImpl, "_recent128_installed", False):
        return
    original_init = AscendSFAImpl.__init__
    original_weights = AscendSFAImpl.process_weights_after_loading
    original_exec = AscendSFAImpl.exec_kv
    original_read = AscendSFAImpl._execute_sparse_flash_attention_process
    original_fused_type = AscendSFAImpl._fused_preprocess_type
    original_prolog = getattr(AscendSFAImpl, "_sfa_preprocess_prolog_v3", None)

    @wraps(original_init)
    def init(impl, *args, **kwargs):
        original_init(impl, *args, **kwargs)
        config = getattr(impl, "vllm_config", None)
        require_explicit_config(getattr(config,"additional_config",None))
        impl._recent128_config = parse_recent128_config(getattr(config, "additional_config", None))
        impl._recent128_runtime = None
        if impl._recent128_config.enabled and impl._recent128_config.execution == "serving":
            if not callable(original_prolog):
                raise RuntimeError("Serving recent128 requires the real _sfa_preprocess_prolog_v3 entry before loading weights")
            from .serving_runtime import preflight_serving
            impl._recent128_serving_preflight = preflight_serving(impl._recent128_config)

    @wraps(original_weights)
    def weights(impl, *args, **kwargs):
        result = original_weights(impl, *args, **kwargs)
        if impl._recent128_config.enabled:
            if impl._recent128_config.execution == "serving":
                if not callable(original_prolog):
                    raise RuntimeError("Serving recent128 requires the real _sfa_preprocess_prolog_v3 entry")
                from .serving_runner import install_runner_hooks
                from .serving_runtime import ServingRuntime
                install_runner_hooks()
                impl._recent128_runtime = ServingRuntime(impl, impl._recent128_config)
            else:
                from .runtime import Recent128Runtime
                impl._recent128_runtime = Recent128Runtime(impl, impl._recent128_config)
        return result

    @wraps(original_fused_type)
    def fused_type(impl):
        # Explicit OFF/ON accuracy experiment only, BEFORE fused weight
        # conversion. Preserve is the default; no runtime type switch after
        # weights have been transformed is allowed.
        if impl._recent128_config.preprocess == "native_accuracy":
            vc = impl.vllm_config
            if vc.model_config.enforce_eager is not True or vc.scheduler_config.max_num_seqs != 1:
                raise RuntimeError("native_accuracy preprocessing is only for the explicit single-request eager experiment")
            return None
        return original_fused_type(impl)

    @wraps(original_exec)
    def exec_kv(impl, kv_no_split, cos, sin, kv_cache, slots, attn_metadata):
        runtime = impl._recent128_runtime
        if runtime is None:
            return original_exec(impl, kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        if impl._recent128_config.execution == "serving":
            raise RuntimeError("Serving recent128 expected its PrologV3 ring producer; native exec_kv fallback is not enabled")
        impl._recent128_write_metadata = attn_metadata
        try:
            return original_exec(impl, kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        finally:
            impl._recent128_write_metadata = None

    @wraps(original_read)
    def read(impl, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
             actual_seq_lengths_query, actual_seq_lengths_key, block_table=None):
        runtime = impl._recent128_runtime
        if runtime is None:
            return original_read(impl, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                                 actual_seq_lengths_query, actual_seq_lengths_key, block_table=block_table)
        return runtime.read(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                            actual_seq_lengths_query, actual_seq_lengths_key, block_table)

    if callable(original_prolog):
        @wraps(original_prolog)
        def prolog(impl, hidden_states, *args, **kwargs):
            runtime = impl._recent128_runtime
            if runtime is None or impl._recent128_config.execution != "serving":
                return original_prolog(impl, hidden_states, *args, **kwargs)
            # attn_metadata is keyword-only in the reviewed VA entry. Row
            # count is static tensor metadata, not a device scalar read.
            with runtime.prolog_scope(kwargs.get("attn_metadata"), hidden_states.shape[0]):
                return original_prolog(impl, hidden_states, *args, **kwargs)

        AscendSFAImpl._sfa_preprocess_prolog_v3 = prolog

    AscendSFAImpl.__init__ = init
    AscendSFAImpl.process_weights_after_loading = weights
    AscendSFAImpl.exec_kv = exec_kv
    AscendSFAImpl._fused_preprocess_type = fused_type
    AscendSFAImpl._execute_sparse_flash_attention_process = read
    AscendSFAImpl._recent128_prolog_hook_available = callable(original_prolog)
    AscendSFAImpl._recent128_installed = True
