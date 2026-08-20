"""A0 -- find the blocks of lettering, and the balloons they sit in.

comic-text-and-bubble-detector, an RT-DETR-v2 r50vd fine-tuned on about 11k
manga, webtoon, manhua and western comic pages, answers in one pass the two
questions the rest of this pipeline used to infer:

    bubble       where a speech balloon is
    text_bubble  a block of lettering inside a balloon
    text_free    a block of lettering laid over artwork

Both were previously guessed, and both guesses are what the reported faults
trace back to. Blocks were built by union-find over pairs of text lines, so
two balloons standing side by side welded into one region and a generously
leaded balloon came apart mid-sentence -- either way the translator was handed
a garbled block and answered in kind. Balloon interiors were recovered by
flood-filling out from the lettering, which fails on a balloon open at one
edge or drawn overlapping another, and the typesetter then fell back to the
bounding box: text set to a rectangle inside a round balloon.

A block is a detection here, not the product of a threshold on the gap
between two lines, and a balloon is a detection too. What the line head is
still better at is *lines*: it marks one bar per line of set lettering at
whatever scale the line is drawn, which is what keeps words in the order the
letterer wrote them. So the two are used for what each is good at -- this
model says which lines belong together and what balloon they sit in, the line
head says where the lines are.

The network is end-to-end and needs no NMS in principle. In practice a
fine-tune emits near-duplicate balloons -- two boxes three pixels apart on the
same balloon, at 0.93 and 0.86 -- so overlapping detections of one class are
still merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import runtime
from ..config import Config
from ..types import Box, RegionKind

runtime.enable_bundled_cuda()

import onnxruntime as ort  # noqa: E402  -- must follow the DLL path fix

# The order the model was trained in. Sound effects are not a class of its
# own, so they keep coming from the line head's silence and the stroke and
# slant measures in `classify`.
BUBBLE = 0
TEXT_IN_BUBBLE = 1
TEXT_FREE = 2

_KINDS = {TEXT_IN_BUBBLE: RegionKind.BALLOON, TEXT_FREE: RegionKind.FREE}


@dataclass(frozen=True)
class TextBlock:
    """One block of lettering the layout model found."""

    box: Box
    kind: RegionKind
    score: float


@dataclass
class Layout:
    """What the layout model saw on one page."""

    blocks: list[TextBlock] = field(default_factory=list)
    bubbles: list[Box] = field(default_factory=list)

    @property
    def boxes(self) -> list[Box]:
        return [block.box for block in self.blocks]

    def bubble_for(self, box: Box, min_cover: float = 0.6) -> Box | None:
        """The balloon a block of lettering sits in, if it sits in one.

        Covering the *lettering* is what decides it, not overlap between the
        two boxes: a balloon is mostly empty space, so an IoU test scores a
        balloon holding one short line no better than the balloon beside it.
        """
        best, best_cover = None, min_cover
        for bubble in self.bubbles:
            ix1, iy1 = max(box.x1, bubble.x1), max(box.y1, bubble.y1)
            ix2, iy2 = min(box.x2, bubble.x2), min(box.y2, bubble.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            cover = inter / box.area if box.area else 0.0
            # A balloon that only just covers the lettering beats a larger one
            # that also covers it: nested balloons in a shout-within-a-shout
            # would otherwise all resolve to the outermost.
            if cover > best_cover or (
                cover >= best_cover and best is not None and bubble.area < best.area
            ):
                best, best_cover = bubble, cover
        return best


def _merge_overlapping(boxes: list[Box], scores: list[float], iou: float) -> list[int]:
    """Indices kept after suppressing weaker overlapping boxes of one class."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    kept: list[int] = []
    for index in order:
        if all(boxes[index].iou(boxes[other]) <= iou for other in kept):
            kept.append(index)
    return kept


class LayoutDetector:
    """Wraps one comic-text-and-bubble-detector session, reused per page."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(cfg.models.layout),
            sess_options=options,
            providers=cfg.models.providers(),
        )

    @property
    def provider(self) -> str:
        return self._session.get_providers()[0]

    def detect(self, image: np.ndarray) -> Layout:
        cfg = self._cfg.detect
        height, width = image.shape[:2]

        # A plain resize, aspect and all. The model was trained on images
        # resized rather than cropped or letterboxed, so squaring the page is
        # what it expects; padding it into a square instead costs recall on
        # the narrow balloons down the edges.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        square = cv2.resize(
            rgb, (cfg.layout_input, cfg.layout_input), interpolation=cv2.INTER_LINEAR
        )
        blob = (square.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        # Post-processing is inside the graph: boxes come back as xyxy in the
        # page pixels named here, not normalised and not in model space.
        labels, boxes, scores = self._session.run(
            None,
            {
                "images": blob,
                "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
            },
        )
        labels, boxes, scores = labels[0], boxes[0], scores[0]

        found: dict[int, tuple[list[Box], list[float]]] = {
            BUBBLE: ([], []),
            TEXT_IN_BUBBLE: ([], []),
            TEXT_FREE: ([], []),
        }
        for label, raw, score in zip(labels, boxes, scores):
            label = int(label)
            if label not in found:
                continue
            floor = (
                cfg.bubble_threshold if label == BUBBLE else cfg.text_threshold
            )
            if score < floor:
                continue
            box = Box(
                max(0, int(round(float(raw[0])))),
                max(0, int(round(float(raw[1])))),
                min(width, int(round(float(raw[2])))),
                min(height, int(round(float(raw[3])))),
            )
            if box.width > 1 and box.height > 1:
                found[label][0].append(box)
                found[label][1].append(float(score))

        layout = Layout()
        bubble_boxes, bubble_scores = found[BUBBLE]
        for index in _merge_overlapping(
            bubble_boxes, bubble_scores, cfg.layout_nms_iou
        ):
            layout.bubbles.append(bubble_boxes[index])

        # The two text classes are suppressed together, not one at a time.
        # They answer the same question -- is there a block of lettering here
        # -- and differ only on whether it sits in a balloon, so a piece of
        # lettering both are unsure about comes back as two boxes on the same
        # spot. Suppressed per class, neither cancels the other: one shout on
        # the reported page arrived as an identical pair, was made into two
        # regions, read twice and lettered twice, one on top of the other.
        #
        # Bubbles stay separate, because a balloon is *meant* to overlap the
        # lettering inside it.
        text_boxes: list[Box] = []
        text_scores: list[float] = []
        text_kinds: list[RegionKind] = []
        for label in (TEXT_IN_BUBBLE, TEXT_FREE):
            label_boxes, label_scores = found[label]
            text_boxes.extend(label_boxes)
            text_scores.extend(label_scores)
            text_kinds.extend([_KINDS[label]] * len(label_boxes))
        for index in _merge_overlapping(
            text_boxes, text_scores, cfg.layout_nms_iou
        ):
            layout.blocks.append(
                TextBlock(text_boxes[index], text_kinds[index], text_scores[index])
            )
        return layout
