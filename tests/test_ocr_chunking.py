"""A block too dense to read whole is read in pieces, and only then.

The recogniser sees 224x224 whatever it is handed, so a block's *effective*
line height -- what the lettering measures once the crop is squared off -- is
what decides whether it can be read at all. Below about 13px the reading stops
partway through and a sentence loses its ending: "...AND IN THE SAME BREATH
ACCEPT THEIR", without the "GIFT!".

Splitting is not free, which is why it is not the default. Read in halves, one
balloon gave "SUDERIOR" for "SUPERIOR" and "WEDE COME" for "WERE SOME": the
recogniser has a sentence to lean on and half a sentence is worth less.
"""

from __future__ import annotations

from mangatl.config import default_config
from mangatl.pipeline.detect import merge_rows as _rows
from mangatl.pipeline.ocr import _chunks, _effective_line_height
from mangatl.types import Box, Region

SHAPE = (4000, 3000)


def _block(lines: int, line_height: int, width: int = 300) -> Region:
    pitch = int(line_height * 1.6)
    boxes = [
        Box(100, 100 + index * pitch, 100 + width, 100 + index * pitch + line_height)
        for index in range(lines)
    ]
    region = Region(id="r00", box=Box.union(boxes))
    region.lines = boxes
    region.source_line_height = float(line_height)
    return region


def test_a_block_that_reads_whole_is_read_whole():
    cfg = default_config().ocr
    region = _block(lines=3, line_height=40)
    assert _effective_line_height(region.box, 40) >= cfg.min_line_pixels
    assert len(_chunks(region, 12, cfg, SHAPE)) == 1


def test_a_dense_block_is_split():
    cfg = default_config().ocr
    region = _block(lines=12, line_height=20)
    assert _effective_line_height(region.box, 20) < cfg.min_line_pixels
    assert len(_chunks(region, 12, cfg, SHAPE)) > 1


def test_the_pieces_of_a_split_block_do_not_share_a_line():
    """Overlapping pieces make the join say the same words twice.

    Left to the full margin, the second piece opened by repeating the end of
    the first -- "...WOULD NEVER" then "WOULD NEVER UTTER SUCH..." -- and the
    joined reading carried both.
    """
    cfg = default_config().ocr
    region = _block(lines=12, line_height=20)
    pieces = _chunks(region, 12, cfg, SHAPE)
    for above, below in zip(pieces, pieces[1:]):
        assert above.y2 <= below.y1, "a line would be shown to two pieces"


def test_a_wide_block_is_not_split_into_one_piece_per_line():
    """Taking lines off a block squeezed by its *width* buys nothing back.

    Uncapped, the search walked down to a line apiece and asked for seven
    reads of one balloon.
    """
    cfg = default_config().ocr
    region = _block(lines=9, line_height=14, width=2000)
    assert len(_chunks(region, 12, cfg, SHAPE)) <= cfg.max_read_pieces


def test_a_block_with_no_line_boxes_is_read_whole():
    cfg = default_config().ocr
    region = Region(id="r00", box=Box(0, 0, 300, 900))
    assert len(_chunks(region, 12, cfg, SHAPE)) == 1


# -- bars on one row are one line -------------------------------------------


def test_bars_side_by_side_are_one_row():
    """The line head can break one row of lettering into several bars.

    "INCONCEIVABLE!" -- white outlined lettering over dark art -- came back as
    a bar over "INCONCEIV" and another over "BLE!". Counted as two lines, the
    block was split *between* them, so the word was cut in half and read as
    "TNANA" and "BLE!". That shipped to the page.
    """
    left = Box(356, 1193, 505, 1206)
    right = Box(553, 1195, 600, 1206)
    assert _rows([left, right]) == [Box.union([left, right])]


def test_bars_of_very_different_heights_are_not_one_row():
    """Which is what keeps this off lettering drawn on a slant.

    A banner reading "Thud Thud Thud" down a diagonal produces bars that do
    overlap vertically -- they step down the page rather than sitting on a
    row -- along with stray fragments of the same effect. Merged, the crop
    stopped holding a whole word, the reading lost the "Thud" that marked it
    a noise, and the banner was translated and lettered over.
    """
    thin = Box(171, 1760, 185, 1765)     # 5px tall
    tall = Box(192, 1761, 200, 1774)     # 13px tall
    assert _rows([thin, tall]) == [thin, tall]


def test_stacked_lines_stay_separate():
    """The gap between two lines of a balloon is a few pixels, and real."""
    top = Box(100, 100, 400, 118)
    bottom = Box(100, 122, 400, 140)
    assert _rows([top, bottom]) == [top, bottom]


def test_a_row_is_never_split_however_wide_it_is():
    """A wide single row is squeezed by its width, not by its line count.

    Splitting it cannot buy back the resolution -- there is nothing to split
    along -- and cutting across it destroys the word.
    """
    cfg = default_config().ocr
    region = Region(id="r00", box=Box(353, 1190, 603, 1209))
    region.lines = [Box(356, 1193, 505, 1206), Box(553, 1195, 600, 1206)]
    region.source_line_height = 12.0
    assert _effective_line_height(region.box.padded(12), 12.0) < cfg.min_line_pixels
    assert len(_chunks(region, 12, cfg, SHAPE)) == 1


def test_a_dense_block_of_real_lines_still_splits():
    """The splitting this guards is still wanted where the lines are real."""
    cfg = default_config().ocr
    region = _block(lines=6, line_height=9, width=900)
    assert len(_chunks(region, 12, cfg, SHAPE)) > 1
