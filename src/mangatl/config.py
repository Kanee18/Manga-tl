"""Typed settings loaded from config.yaml.

Every path in the file is interpreted relative to the project root, so the app
runs the same whatever the working directory is. `load_config` is cached, and
`Config.model_copy(update=...)` is how the GUI applies a per-run override
without touching the file.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .languages import DEFAULT_PAIR, pair_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Choices made in the settings panel land here rather than in config.yaml.
# Every threshold in config.yaml carries the measurement that set it in a
# comment beside it, and writing that file back through a YAML dumper would
# discard all of them -- which is most of what makes the file worth having.
OVERRIDES_PATH = PROJECT_ROOT / "settings.local.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


class ModelsConfig(BaseModel):
    detector: Path
    layout: Path
    ocr: Path
    inpaint: Path
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @field_validator("detector", "layout", "ocr", "inpaint", mode="before")
    @classmethod
    def _abs(cls, v: Any) -> Path:
        return _resolve(v)

    def providers(self) -> list[str]:
        """onnxruntime execution providers, most preferred first."""
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        if self.device == "cpu":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "cuda":
            raise RuntimeError(
                "device: cuda was requested but onnxruntime exposes no CUDA "
                f"provider (available: {sorted(available)}). Install "
                "onnxruntime-gpu, or set models.device to cpu."
            )
        return ["CPUExecutionProvider"]


class DetectConfig(BaseModel):
    input_size: int = 1024
    mask_threshold: float = 0.30
    line_threshold: float = 0.30
    line_threshold_weak: float = 0.12
    min_area: int = 260
    merge_gap_ratio: float = 2.2
    gutter_ratio: float = 1.2
    max_block_area_ratio: float = 0.22
    reading_order: Literal["rtl", "ltr"] = "rtl"
    # The layout model is exported for a plain square resize; changing this
    # does not trade quality for speed, it feeds the graph the wrong shape.
    layout_input: int = 640
    text_threshold: float = 0.45
    bubble_threshold: float = 0.50
    layout_nms_iou: float = 0.50
    recover_unclaimed: bool = True
    recover_max_stroke_ratio: float = 1.95
    recover_max_slant: float = 14.0


class OCRConfig(BaseModel):
    # vision_fp16.onnx is the published model; vision_int4.onnx is a third the
    # size for about 0.002 more character error.
    vision: str = "vision_fp16.onnx"
    # The vocabulary is one entry per character, so this is a character budget
    # and not a word one. A long balloon runs to about 90.
    max_chars: int = 384
    repetition_penalty: float = 1.2
    max_character_run: int = 12
    # Air around the lettering in the crop, as a fraction of the block height.
    block_margin_ratio: float = 0.16
    min_block_margin: int = 12
    # How tall the source lettering must land once the crop is squared
    # off to 224px, or the block is read in pieces instead.
    min_line_pixels: float = 13.0
    max_read_pieces: int = 3
    min_chars: int = 2


class ClassifyConfig(BaseModel):
    flat_bg_std: float = 18.0
    # How far the stroke must sit from the ink before it is worth drawing.
    min_outline_contrast: float = 40.0


class TranslateConfig(BaseModel):
    # Which way round the translation goes, as "<source>-<target>". Drives the
    # prompt, the repair guidance, the still-untranslated guard and how
    # recognised lines are joined. See mangatl.languages.
    pair: str = DEFAULT_PAIR
    host: str = "http://localhost:11434"
    model: str = "gemma4:e2b"
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = 0
    num_ctx: int = 8192
    # Generous, because a 9B model that does not fit entirely in VRAM spills
    # layers to the CPU and a whole-page request can take two or three
    # minutes. A timeout here is not a small delay -- it throws the page's
    # only context-carrying request away.
    timeout: float = 600.0
    max_retries: int = 3
    # How many lines are worth a request of their own after the page batch
    # and the corrective pass have both failed on them.
    max_single_retries: int = 3
    page_batch: bool = True
    # Sound effects drawn into the artwork are left as drawn by default:
    # erasing one takes a piece of the drawing with it, and the replacement
    # would be the comic face where the artist lettered by hand.
    translate_sfx: bool = False
    glossary: dict[str, str] = Field(default_factory=dict)


class InpaintConfig(BaseModel):
    method: Literal["auto", "lama", "fill"] = "auto"
    dilate: int = 5
    lama_size: int = 512
    context_ratio: float = 0.35
    # How far into a hole the inpainter is asked to invent, in page pixels.
    # Past this the block rectangle is abandoned and the glyphs are followed
    # instead. See `pipeline.inpaint._block_mask`.
    max_hole_depth: float = 150.0


class TypesetConfig(BaseModel):
    font: Path
    font_bold: Path
    font_italic: Path
    uppercase: bool = True
    min_font_size: float = 8.0
    max_font_size: float = 44.0
    line_spacing: float = 1.15
    padding_ratio: float = 0.10
    hyphenate: bool = True
    max_growth_over_source: float = 1.15
    outline_width_ratio: float = 0.20

    @field_validator("font", "font_bold", "font_italic", mode="before")
    @classmethod
    def _abs(cls, v: Any) -> Path:
        return _resolve(v)


class OutputConfig(BaseModel):
    suffix: str = "_ID"
    format: Literal["png", "jpg"] = "png"
    jpeg_quality: int = 95
    debug: bool = False


class Config(BaseModel):
    models: ModelsConfig
    detect: DetectConfig = Field(default_factory=DetectConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    inpaint: InpaintConfig = Field(default_factory=InpaintConfig)
    typeset: TypesetConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    def check_assets(self) -> list[str]:
        """Missing files that would make a run fail, as readable messages."""
        problems: list[str] = []
        for label, path in [
            ("text detector", self.models.detector),
            ("layout detector", self.models.layout),
            ("recogniser", self.models.ocr / self.ocr.vision),
            ("inpainting model", self.models.inpaint),
            ("lettering font", self.typeset.font),
        ]:
            if not path.exists():
                problems.append(f"{label} not found at {path}")
        return problems


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """`overlay` wins, section by section rather than section-for-section."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | None = None, overrides: bool = True) -> Config:
    """Read config.yaml, with settings.local.yaml layered over it."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"no config file at {config_path}")
    raw = _read_yaml(config_path)
    if overrides and path is None and OVERRIDES_PATH.exists():
        raw = _merge(raw, _read_yaml(OVERRIDES_PATH))
    return Config.model_validate(raw)


def load_overrides() -> dict[str, Any]:
    """Whatever the settings panel saved last, or an empty mapping."""
    return _read_yaml(OVERRIDES_PATH) if OVERRIDES_PATH.exists() else {}


def save_overrides(values: dict[str, Any]) -> None:
    """Persist settings-panel choices, leaving config.yaml untouched."""
    default_config.cache_clear()
    with OVERRIDES_PATH.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Written by MangaTL's settings panel. Values here win over\n"
            "# config.yaml; delete this file to go back to the defaults.\n"
        )
        yaml.safe_dump(values, fh, sort_keys=True, allow_unicode=True)


@functools.lru_cache(maxsize=4)
def default_config() -> Config:
    """The process-wide config, read once."""
    return load_config()
