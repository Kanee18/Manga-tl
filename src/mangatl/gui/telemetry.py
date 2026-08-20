"""What the machine is doing while a chapter runs.

A page takes twenty seconds or so and a chapter takes twenty of those, which
is long enough that the question stops being "is it working" and becomes "is
it working *hard*, and on what". Four numbers answer it: the processor, system
memory, the card's memory, and the power the card is drawing. Between them
they say whether the models are on the GPU at all, whether VRAM is about to
run out, and whether a stall is real work or a hang.

VRAM is the one that changes what a user does. Detection, recognition and
inpainting each hold a session, the translator holds a model inside Ollama,
and on an 8 GB card those together are most of it -- so a run that spills to
the CPU is not a mystery slowdown, it is a number climbing to the top of the
bar and staying there.

The GPU is read through NVML, the library `nvidia-smi` itself wraps. Reading
it costs microseconds, where shelling out to `nvidia-smi` costs a process
launch every sample -- but that fallback is kept, because a machine with a
working driver and no Python binding should still show its card. Neither being
available is not an error: an AMD or Apple machine runs the pipeline on the
CPU perfectly well, and simply has no card to report.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import psutil
from PyQt6.QtCore import QThread, pyqtSignal

# Sampling cadence. Fast enough that a stage change is visible, slow enough
# that the strip is readable rather than flickering.
INTERVAL_MS = 1000

_MIB = 1024 * 1024


@dataclass(frozen=True)
class Gauge:
    """One measured quantity against whatever counts as full for it."""

    used: float
    total: float
    unit: str

    @property
    def fraction(self) -> float:
        return max(0.0, min(1.0, self.used / self.total)) if self.total else 0.0

    @property
    def text(self) -> str:
        if self.unit == "%":
            return f"{self.used:.0f}%"
        if self.unit == "W":
            return f"{self.used:.0f} W"
        if self.total >= 1024:  # mebibytes, shown as gibibytes past a gigabyte
            return f"{self.used / 1024:.1f} / {self.total / 1024:.0f} GB"
        return f"{self.used:.0f} / {self.total:.0f} {self.unit}"

    @property
    def detail(self) -> str:
        if self.unit == "%":
            return f"{self.used:.1f}% of capacity"
        if self.unit == "W":
            return f"{self.used:.1f} W of a {self.total:.0f} W limit"
        return f"{self.used:.0f} of {self.total:.0f} {self.unit} in use"


@dataclass(frozen=True)
class Reading:
    """One sample of the machine. GPU fields are None where there is no card."""

    cpu: Gauge
    memory: Gauge
    vram: Gauge | None = None
    power: Gauge | None = None
    gpu_util: Gauge | None = None
    gpu_name: str = ""


class _Nvml:
    """NVML through its Python binding. The cheap path, and the detailed one."""

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(self._handle)
        self.name = name if isinstance(name, str) else name.decode()
        # Not every card reports a limit, and one that does not still reports
        # draw. Fall back to the board's default so the bar has a scale.
        self._limit = 0.0
        for lookup in (
            pynvml.nvmlDeviceGetEnforcedPowerLimit,
            pynvml.nvmlDeviceGetPowerManagementLimit,
        ):
            try:
                self._limit = lookup(self._handle) / 1000.0
                break
            except pynvml.NVMLError:
                continue

    def sample(self) -> tuple[Gauge | None, Gauge | None, Gauge | None]:
        memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        vram = Gauge(memory.used / _MIB, memory.total / _MIB, "MiB")

        power = None
        try:
            draw = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            power = Gauge(draw, self._limit or draw or 1.0, "W")
        except self._nvml.NVMLError:
            pass

        util = None
        try:
            util = Gauge(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu, 100.0, "%")
        except self._nvml.NVMLError:
            pass
        return vram, power, util

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 -- shutting down is best effort
            pass


class _Smi:
    """`nvidia-smi`, for a driver present without the Python binding."""

    _FIELDS = "name,memory.used,memory.total,power.draw,power.limit,utilization.gpu"

    def __init__(self) -> None:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi is not on PATH")
        self.name = ""
        if self._query() is None:
            raise RuntimeError("nvidia-smi returned nothing usable")

    def _query(self) -> list[str] | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self._FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=4,
                # Without this a console window flashes on top of the app once
                # a second on Windows, which is worse than having no readout.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        line = out.stdout.strip().splitlines()
        if not line:
            return None
        parts = [field.strip() for field in line[0].split(",")]
        return parts if len(parts) >= 6 else None

    def sample(self) -> tuple[Gauge | None, Gauge | None, Gauge | None]:
        parts = self._query()
        if parts is None:
            return None, None, None
        self.name = parts[0]

        def number(text: str) -> float | None:
            try:
                return float(text)
            except ValueError:  # "[N/A]" on cards that do not report it
                return None

        used, total = number(parts[1]), number(parts[2])
        draw, limit = number(parts[3]), number(parts[4])
        util = number(parts[5])
        return (
            Gauge(used, total, "MiB") if used is not None and total else None,
            Gauge(draw, limit or draw or 1.0, "W") if draw is not None else None,
            Gauge(util, 100.0, "%") if util is not None else None,
        )

    def close(self) -> None:
        return


def _gpu() -> _Nvml | _Smi | None:
    for backend in (_Nvml, _Smi):
        try:
            return backend()
        except Exception:  # noqa: BLE001 -- absence is the normal case, not a fault
            continue
    return None


class TelemetryWorker(QThread):
    """Samples the machine on its own thread and hands readings to the window.

    On a thread rather than a timer because the fallback shells out, and a
    hundred milliseconds of subprocess on the GUI thread once a second is a
    visible stutter in the preview.
    """

    sampled = pyqtSignal(object)  # Reading

    def __init__(self, interval_ms: int = INTERVAL_MS) -> None:
        super().__init__()
        self._interval = interval_ms
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        gpu = _gpu()
        # The first call establishes the baseline the next one measures
        # against; asking for it here means the first reading shown is real
        # rather than whatever the process averaged since it started.
        psutil.cpu_percent(interval=None)
        try:
            while self._running:
                memory = psutil.virtual_memory()
                vram = power = util = None
                if gpu is not None:
                    try:
                        vram, power, util = gpu.sample()
                    except Exception:  # noqa: BLE001 -- a card can go away
                        gpu, vram, power, util = None, None, None, None
                self.sampled.emit(
                    Reading(
                        cpu=Gauge(psutil.cpu_percent(interval=None), 100.0, "%"),
                        memory=Gauge(
                            (memory.total - memory.available) / _MIB,
                            memory.total / _MIB,
                            "MiB",
                        ),
                        vram=vram,
                        power=power,
                        gpu_util=util,
                        gpu_name=getattr(gpu, "name", ""),
                    )
                )
                # Slept in slices so quitting the app does not wait out a
                # whole interval before the thread notices.
                for _ in range(max(1, self._interval // 50)):
                    if not self._running:
                        break
                    self.msleep(50)
        finally:
            if gpu is not None:
                gpu.close()
