"""The per-page stages, in the order a page moves through them.

Importing this package applies the CUDA DLL fix before any onnxruntime session
is built, so importing a stage is always enough.

Stages are not re-exported here on purpose: each one pulls in a heavy
dependency (onnxruntime, skia), and the GUI imports them lazily so
the window is on screen before any model loads.
"""

from .. import runtime

runtime.enable_bundled_cuda()

__all__ = ["runtime"]
