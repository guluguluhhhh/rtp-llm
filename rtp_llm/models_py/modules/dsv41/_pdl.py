"""PDL policy for V4.1 opt-in sites; external libraries keep their own policy."""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _device_supports_pdl(index: int) -> bool:
    return torch.cuda.get_device_capability(index)[0] >= 9


def enabled(device: torch.device) -> bool:
    # Keep the override live for eager A/B runs. Set it before JIT warmup/capture;
    # replay retains the launch attributes recorded in the graph.
    if (
        os.environ.get("DSV41_ENABLE_PDL", "1") != "1"
        or os.environ.get("DISABLE_PDL", "0") == "1"
        or device.type != "cuda"
        or torch.version.hip is not None
    ):
        return False
    index = device.index
    return _device_supports_pdl(torch.cuda.current_device() if index is None else index)


def launch_options(tensor: torch.Tensor) -> dict:
    if enabled(tensor.device):
        return {"USE_PDL": True, "launch_pdl": True}
    # launch_pdl is a CUDA backend option; omit it for the fallback backend.
    return {"USE_PDL": False}


@triton.jit
def pdl_sync(USE_PDL: tl.constexpr):
    if USE_PDL:
        # Wait before reading predecessor outputs. Qualified kernels may
        # preload immutable weights before this point.
        # Match DeepInfer's default: signal on kernel exit. Early signaling is
        # reserved for producers with an explicitly qualified overlap window.
        tl.extra.cuda.gdc_wait()
