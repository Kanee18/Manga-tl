"""A3a -- Baberu OCR, the recogniser that reads a block of lettering whole.

A DINOv2-base encoder over a 224x224 crop, then a six-layer decoder that
writes the text out a character at a time. Trained on comic speech balloons in
Japanese, Chinese and English, which is the difference that matters here: a
general line recogniser has never seen a hand-lettered comic face, and reads
one about as well as it reads handwriting.

Measured against the PP-OCR line recogniser this replaces, over three pages of
one chapter:

    PP-OCR                                          Baberu
    Whoever dio gou inherit you hu Ikish bodu       Whoever did you inherit your hulkish body
      and hideous face from?                          and hideous face from?
    ANDTHEN ABANDONED METOFIND WAYSOLT              AND THEN ABANDONED ME TO FIND WAYS OUT
      OFTHE GROWING MONEY BOBLEMS                     OF THE GROWING MONEY PROBLEMS.
    FATHERI HERE'ARE OUI'GOING?!                    FATHER, WHERE ARE YOU GOING?!
    AND WITH ENOUGH Q L CAN STEAL ANOTHER'S BODY    AND WITH ENOUGH QI, I CAN STEAL ANOTHER'S BODY.
    HAVE T YOUA WAY THIEN                           HAVE IT YOUR WAY THEN!

Note the full stops and exclamation marks. The old readings lost them, and
losing them costs more than it looks: a shout set as a statement reads wrong
on the page, and the translator was being asked to guess the punctuation back.

It reads a *block*, not a line, so everything the line path needed -- cutting
words apart at gaps in the glyph mask because the Chinese recogniser would not
emit spaces, ordering the pieces back into lines, recognising each word on its
own -- is gone. The line breaks come back in the output, so hyphenated breaks
can still be healed.

Two things about the crop matter, and both were measured rather than guessed:

* Frame the *lettering*, not the balloon. 224x224 is the whole budget, so
  every pixel of balloon around the text is resolution the glyphs do not get.
  Fed balloon crops, the longest lines came back cut off mid-word -- "JEERED
  AT M", "BEFORE THAT, WOULD" -- and two blocks sharing one balloon came back
  with identical text. Fed the block, the same lines came back whole.
* Leave a margin. A crop flush to the glyphs clips ascenders and the model
  reads the top row as the row below it.

The decode is greedy with a repetition penalty and a cap on how long one
character may repeat, which is what the model was published with. Adapted from
the reference `onnx_infer.py` in genshiai-daichi/baberu-ocr (Apache-2.0).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .. import runtime
from ..config import Config

runtime.enable_bundled_cuda()

import onnxruntime as ort  # noqa: E402  -- must follow the DLL path fix

# DINOv2 over ImageNet statistics, no centre crop: the whole block is resized
# into the square, so a wide block is squashed rather than cropped.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# The square the encoder sees, whatever the crop was.
SIDE = 224

_LAYERS = 6
_PAST = [f"past_k{i}" for i in range(_LAYERS)] + [f"past_v{i}" for i in range(_LAYERS)]

_PAD, _BOS, _EOS, _UNK = 0, 1, 2, 3
_SPECIAL = 4

# Characters that may not repeat forever. A run of one letter is the failure
# mode of a greedy decoder that has lost the image, and without a cap it fills
# the whole budget with it.
_ELONGATION = "ーｰ〜~"


class Recogniser:
    """The three Baberu graphs, loaded once and run per block."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        directory = Path(cfg.models.ocr)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        providers = cfg.models.providers()

        def load(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(directory / name), sess_options=options, providers=providers
            )

        self._vision = load(cfg.ocr.vision)
        self._prefill = load("decoder_prefill_int8.onnx")
        self._step = load("decoder_step_int8.onnx")

        charset = json.loads(
            (directory / "vocab.json").read_text(encoding="utf-8")
        )
        self._chars = {index + _SPECIAL: ch for index, ch in enumerate(charset)}
        self._content = {
            index + _SPECIAL
            for index, ch in enumerate(charset)
            if len(ch) == 1
            and ch not in _ELONGATION
            and unicodedata.category(ch)[0] in "LN"
        }

    @property
    def provider(self) -> str:
        return self._vision.get_providers()[0]

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """The lettering in one BGR crop, and how sure the model was of it."""
        if crop.size == 0:
            return "", 0.0
        cfg = self._cfg.ocr
        embeds = self._vision.run(
            ["vision_embeds"], {"pixel_values": _prepare(crop)}
        )[0]
        outputs = self._prefill.run(
            None,
            {
                "vision_embeds": embeds,
                "input_ids": np.array([[_BOS]], dtype=np.int64),
            },
        )
        logits, past = outputs[0][0, -1].astype(np.float64), outputs[1:]

        position = embeds.shape[1] + 1
        seen: set[int] = {_BOS}
        tokens: list[int] = []
        confidences: list[float] = []

        for _ in range(cfg.max_chars):
            # Penalising what has already been written, in both directions of
            # sign so a negative logit is pushed down rather than up.
            for token in seen:
                score = logits[token]
                logits[token] = (
                    score * cfg.repetition_penalty
                    if score < 0
                    else score / cfg.repetition_penalty
                )
            if tokens and tokens[-1] in self._content:
                run = 0
                for token in reversed(tokens):
                    if token != tokens[-1]:
                        break
                    run += 1
                if run >= cfg.max_character_run:
                    logits[tokens[-1]] = -np.inf

            chosen = int(np.argmax(logits))
            confidences.append(_softmax_at(logits, chosen))
            if chosen == _EOS:
                break
            tokens.append(chosen)
            seen.add(chosen)
            if len(tokens) >= cfg.max_chars:
                break

            feed: dict[str, np.ndarray] = {
                "input_ids": np.array([[chosen]], dtype=np.int64),
                "position_ids": np.array([[position]], dtype=np.int64),
            }
            feed.update(dict(zip(_PAST, past)))
            outputs = self._step.run(None, feed)
            logits, past = outputs[0][0, -1].astype(np.float64), outputs[1:]
            position += 1

        text = "".join(self._chars.get(t, "") for t in tokens if t >= _SPECIAL)
        # The model writes Japanese-width forms for latin letters and
        # ellipses when the lettering leans that way -- "ＨＡＨ…" for "HAH...".
        # Left alone they reach the translator as characters it has to guess
        # at, and the typesetter as glyphs the comic face has no room for.
        text = unicodedata.normalize("NFKC", text)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return text, confidence


def _prepare(crop: np.ndarray) -> np.ndarray:
    """One BGR crop as the tensor the vision encoder expects."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    square = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
    normalised = (square.astype(np.float32) / 255.0 - _MEAN) / _STD
    return normalised.transpose(2, 0, 1)[None]


def _softmax_at(logits: np.ndarray, index: int) -> float:
    """The probability the decoder gave the character it chose."""
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    total = float(exponentiated.sum())
    return float(exponentiated[index] / total) if total else 0.0
