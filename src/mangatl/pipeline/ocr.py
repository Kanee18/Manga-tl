"""A3 -- read the English lettering in every region.

One request per block of lettering, not per line and not per word. The
recogniser behind it (`pipeline.baberu`) was trained on comic balloons and
reads a whole block at once, which removes the two things the previous line
recogniser needed and neither of which ever quite worked:

* words no longer have to be cut apart at gaps in the glyph mask. That existed
  because the bundled Chinese PP-OCR model emits spaces only intermittently in
  English, so a line came back as "SHEIS BARELY CONSCIOUS ANDAT TIMES" and the
  spaces had to be recovered from the mask instead;
* the pieces no longer have to be sorted back into reading order afterwards,
  which is where "AT THIS RATE--" once became "AT SIHL RATE--".

Two manga-specific things still happen after recognition:

* **Hyphenated breaks are rejoined.** Balloons are narrow, so letterers break
  words across lines constantly -- the sample page alone has `CONVUL-/SIONS.`,
  `CON-/SCIOUS` and `HALLUCI-/NATING.` Passed through as written, the
  translator is asked to translate three words that do not exist, and it will
  cheerfully invent Indonesian for them. The recogniser returns the letterer's
  own line breaks, so the healing has real line ends to work with.

* **Loose letters are refused.** Vertical sound effects and rotated lettering
  come back as scattered capitals, and a region that reads as noise is left as
  artwork rather than erased and lettered with nonsense.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import Config, OCRConfig
from ..languages import CJK, pair_for
from ..types import Box, Region, RegionStatus
from .baberu import Recogniser, SIDE
from .detect import merge_rows

# A trailing single hyphen is a word broken across lines. A double hyphen is
# the letterer's dash -- "AT THIS RATE--" -- and joins nothing.
_SOFT_BREAK = re.compile(r"(?<![-‐-―])-$")

# A line with nothing readable in it. Kana and han count as readable: the
# original pattern was `^[^0-9A-Za-z]+$`, which is every Japanese and Chinese
# line ever recognised, so a CJK page came back entirely "nothing but
# punctuation" and not one balloon was translated.
_JUNK = re.compile(r"^[^0-9A-Za-z぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]+$")

_LETTER = re.compile(r"[A-Za-z]")


def _join(lines: list[str], spaced: bool = True) -> str:
    """One string from several lettered lines.

    Latin lettering breaks across lines with a space between the pieces, and
    sometimes a hyphen to heal. Japanese and Chinese are written without
    spaces at all, so joining their lines with one inserts a gap the letterer
    never put there -- and the translator is then reading a sentence with a
    word break in the middle of a word.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif spaced and _SOFT_BREAK.search(out):
            out = out[:-1] + line
        elif spaced:
            out = f"{out} {line}"
        else:
            out = out + line
    return re.sub(r"\s+", " ", out).strip() if spaced else out.strip()


# Single letters that are words in their own right. Counting these as noise
# threw away real dialogue: "I AM A" is two thirds isolated letters and was
# being dropped, which cost a balloon the middle of its own sentence.
_STANDALONE_LETTERS = frozenset({"i", "a", "o"})


def _is_gibberish(text: str, spaced: bool = True) -> bool:
    """Whether a reading is loose letters rather than words.

    Vertical sound effects and rotated lettering come back from the recogniser
    as scattered capitals -- "T R D N V", "0 INOY G WOM". Passed on, the
    translator does what it is asked and invents Indonesian for them, and a
    balloon that should have been left as artwork is erased and lettered with
    nonsense.

    Only letters that are *not* words on their own are counted, so ordinary
    dialogue keeps its "I" and "A" however short the line is.
    """
    # Only meaningful for a script that separates words. Japanese and Chinese
    # come back as one token however long the line is, so every reading would
    # pass this test trivially -- and a rotated CJK sound effect is not loose
    # letters anyway, it is characters that happen to be wrong.
    if not spaced:
        return False
    tokens = text.split()
    if len(tokens) < 3:
        return False
    stray = sum(
        1
        for token in tokens
        if len(bare := token.strip(".,!?'\"-").lower()) <= 1
        and bare not in _STANDALONE_LETTERS
    )
    return stray / len(tokens) >= 0.5


def _effective_line_height(box: Box, line_height: float) -> float:
    """How tall the source lettering lands once the crop is squared off.

    The recogniser sees 224x224 whatever it is given, so this -- not the size
    of the lettering on the page -- is what decides whether it can read it.
    """
    longest = max(box.width, box.height)
    return line_height * SIDE / longest if longest else 0.0


def _chunks(
    region: Region, margin: int, cfg: OCRConfig, shape: tuple[int, int]
) -> list[Box]:
    """The crops one block is read in, top to bottom.

    Normally one: a block read whole is read better, because the recogniser
    has the whole sentence to lean on. "DON'T YOU DARE SPEAK MY SUPERIOR
    MAIDEN'S NAME!" read whole; read in halves the same block gave "SUDERIOR",
    and "WERE SOME LESSER LADY" became "WEDE COME LESSER LADY".

    A block too dense to survive being squeezed into the square is the
    exception. Measured over one chapter, blocks whose lettering lands above
    about 11px come back whole, and below it the reading stops partway --
    "...AND IN THE SAME BREATH ACCEPT THEIR", with the "GIFT!" that makes the
    sentence a sentence missing. Those are split at line boundaries into as
    few pieces as will read, and joined back up afterwards.
    """
    box = region.box.padded(margin, within=shape)
    lines = merge_rows(region.lines)
    if len(lines) < 2 or not region.source_line_height:
        return [box]
    if _effective_line_height(box, region.source_line_height) >= cfg.min_line_pixels:
        return [box]

    # The most lines that still read, found by trying the whole block and
    # taking a line off until it fits.
    count = len(lines)
    while count > 1:
        trial = Box.union(lines[:count]).padded(margin, within=shape)
        if _effective_line_height(trial, region.source_line_height) >= cfg.min_line_pixels:
            break
        count -= 1

    # Capped. A wide block is squeezed by its width, and taking lines off it
    # never buys back the resolution, so the search above walks all the way
    # down to one line per piece and asks for seven reads of one balloon. Past
    # a few pieces each is so small a slice that the recogniser has no
    # sentence left to lean on, and the cost is a whole read apiece.
    pieces = min(cfg.max_read_pieces, max(2, -(-len(lines) // count)))
    per = -(-len(lines) // pieces)
    groups = [lines[at : at + per] for at in range(0, len(lines), per)]

    boxes: list[Box] = []
    for index, group in enumerate(groups):
        bounds = Box.union(group)
        # Vertical margin is clamped to the gap on either side, so no line is
        # shown to two pieces. Without that the second piece opens by
        # repeating the end of the first: "...WOULD NEVER" then "WOULD NEVER
        # UTTER SUCH...", and joining them said it twice.
        above = margin if index == 0 else min(
            margin, max(0, (bounds.y1 - groups[index - 1][-1].y2) // 2)
        )
        below = margin if index == len(groups) - 1 else min(
            margin, max(0, (groups[index + 1][0].y1 - bounds.y2) // 2)
        )
        height, width = shape
        boxes.append(
            Box(
                max(0, bounds.x1 - margin),
                max(0, bounds.y1 - above),
                min(width, bounds.x2 + margin),
                min(height, bounds.y2 + below),
            )
        )
    return boxes


class TextReader:
    """Wraps one recogniser. Build once, reuse for every region."""

    def _in_source_script(self, line: str) -> str:
        """`line` with characters the source language does not use removed.

        The recogniser reads Japanese, Chinese and English, and is told none
        of the three in advance. On an English page it therefore returns kana
        now and again -- sometimes a misreading of damaged lettering, and
        sometimes a perfectly correct reading of a Chinese banner drawn into
        the artwork. Neither is the script being translated, and both reached
        the page: `それでも` was read on an English page and
        lettered back as "Tidak...".

        Stripping rather than refusing the whole line, because a single stray
        character in an otherwise good reading is damage worth repairing and
        not a reason to lose the balloon. What is left of the line decides
        whether anything was really there.

        Only guarded in this direction. A Latin word in a Japanese page is
        usually a name -- ABYDOS, KANT -- and dropping those would be a
        different bug of the same shape.
        """
        if self._source.is_cjk or not CJK.search(line):
            return line
        return CJK.sub("", line).strip()

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._recogniser = Recogniser(cfg)
        self._source = pair_for(cfg.translate.pair).source
        self._spaced = self._source.spaced

    def read(self, image: np.ndarray, region: Region) -> None:
        """Fill in text_en, ocr_lines and ocr_confidence for one region."""
        cfg = self._cfg.ocr
        if region.box.crop(image).size == 0:
            region.fail(RegionStatus.FAILED_OCR, "empty crop")
            return

        # Framed on the lettering with air around it. The margin scales with
        # the block because a caption set at 12px and a shout set at 90px need
        # the same margin in *ems*, not in pixels, and the floor keeps a small
        # block from being framed so tightly its ascenders are clipped.
        margin = max(
            cfg.min_block_margin, int(round(region.box.height * cfg.block_margin_ratio))
        )
        readings: list[str] = []
        scores: list[float] = []
        for crop in _chunks(region, margin, cfg, image.shape[:2]):
            text, confidence = self._recogniser.read(crop.crop(image))
            if text.strip():
                readings.append(text)
                scores.append(confidence)

        lines = [
            line
            for text in readings
            for line in text.splitlines()
            if line.strip()
        ]
        confidence = float(np.mean(scores)) if scores else 0.0
        region.ocr_lines = [
            line.strip() for line in lines if not _JUNK.match(line.strip())
        ]
        region.ocr_lines = [
            self._in_source_script(line) for line in region.ocr_lines
        ]
        region.ocr_lines = [line for line in region.ocr_lines if line]
        region.text_en = _join(region.ocr_lines, self._spaced)
        region.ocr_confidence = confidence

        if not region.text_en:
            region.fail(
                RegionStatus.SKIPPED_EMPTY,
                "nothing but punctuation" if lines else "no text recognised",
            )
            return
        if len(region.text_en) < cfg.min_chars:
            region.fail(RegionStatus.SKIPPED_EMPTY, f"too short: {region.text_en!r}")
            return
        if _is_gibberish(region.text_en, self._spaced):
            region.fail(RegionStatus.SKIPPED_EMPTY, f"unreadable: {region.text_en!r}")
            return
        region.status = RegionStatus.READ
