"""Process-wide setup that has to happen before any model session is created.

Import this module before `onnxruntime` builds a session. `mangatl.pipeline`
does it at package import, so nothing else needs to remember to.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path


@functools.lru_cache(maxsize=1)
def enable_bundled_cuda() -> bool:
    """Put torch's CUDA 12 / cuDNN 9 libraries on the DLL search path.

    onnxruntime-gpu loads `cublasLt64_12.dll`, `cudnn64_9.dll` and friends by
    name, so they have to be somewhere Windows will look. A machine with the
    cu12x torch wheels already has every one of them, but inside
    `torch/lib`, which is not on PATH and not a directory onnxruntime
    searches. Without this the CUDA provider fails to register and the session
    silently falls back to CPU -- the run still works, roughly twenty times
    slower, with nothing in the output to say why.

    Returns True when a directory was added. Safe to call repeatedly.
    """
    if sys.platform != "win32":
        return False
    try:
        import torch
    except ImportError:
        return False

    lib = Path(torch.__file__).parent / "lib"
    if not (lib / "cublasLt64_12.dll").exists():
        return False

    os.add_dll_directory(str(lib))
    # Some loader paths consult PATH rather than the added directory list.
    os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"
    return True


def describe_device() -> str:
    """A one-line summary of what the models will actually run on."""
    enable_bundled_cuda()
    try:
        import onnxruntime as ort
    except ImportError:
        return "onnxruntime not installed"
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        try:
            import torch

            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass
        return "CUDA"
    return "CPU"
