"""Reject an environment-only ON request that this overlay does not consume."""
from collections.abc import Mapping
import os


UNSUPPORTED_ACTIVATION_ENV=("SFA_RECENT128_CONFIG","SFA_RECENT128_LIB_DIR")


def require_explicit_config(additional_config,environ=None):
    """Never translate undocumented environment JSON into framework config.

    A real OFF/ON additional_config remains authoritative. If only the legacy
    environment request exists, fail before the wrapper can silently run OFF.
    No config, environment, plugin selection or installed package is modified.
    """
    environ=os.environ if environ is None else environ
    names=[name for name in UNSUPPORTED_ACTIVATION_ENV if name in environ]
    configured=isinstance(additional_config,Mapping) and "sfa_recent128" in additional_config
    if names and not configured:
        raise RuntimeError(
            "recent128 was requested only through unsupported environment variables "
            +", ".join(names)+". This overlay reads VllmConfig.additional_config['sfa_recent128']; "
            "these environment variables do not activate it. Pass the explicit framework config "
            "via --additional-config: serving uses window=128, execution='serving', "
            "preprocess='preserve', reader='single_pass', single_pass_kernel='batched_v4', "
            "source_root, library_dir, diagnostics_path and the actual max_query_chunk. "
            "The separate execution='accuracy_eager' mode (including kv_split4) is an isolated "
            "single-request validation path. No environment fallback was applied."
        )
