"""Tests for lettering, which is where a defect is most visible to a reader."""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline.typeset import (
    Typesetter, _breakable, _covered, _drawable_area,
)
from mangatl.types import Box, Region, RegionStatus


def _latin_config():
    """The English-source config, whatever this machine happens to be set to.

    `default_config()` reads settings.local.yaml, so a developer who has
    switched the app to Japanese changes what these tests measure: source-
    matched sizing does not apply to a vertical source, and the assertions
    below are about source-matched sizing.
    """
    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "en-id"
    return cfg


@pytest.fixture(scope="module")
def typesetter() -> Typesetter:
    return Typesetter(_latin_config())


def _balloon(width: int = 200, height: int = 160) -> Region:
    """A region in an elliptical balloon, as classification would produce."""
    centre_x, centre_y = 400, 400
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    polygon = np.stack(
        [
            centre_x + (width / 2) * np.cos(angles),
            centre_y + (height / 2) * np.sin(angles),
        ],
        axis=1,
    ).astype(np.int32)
    region = Region(
        id="r00",
        box=Box(centre_x - 60, centre_y - 40, centre_x + 60, centre_y + 40),
        polygon=polygon,
        source_line_height=12.0,
        status=RegionStatus.TRANSLATED,
    )
    return region


def test_text_is_shrunk_before_it_is_broken(typesetter):
    """Hyphenation can make anything fit, so it must not be offered first.

    Offering it during the size search stopped the search at the largest size
    and produced "NO-NA REIRI-N!" where a slightly smaller size held
    "NONA REIRIN!" whole.
    """
    region = _balloon()
    region.text_id = "Nona Reirin!"
    layout = typesetter.layout(region)
    assert layout is not None
    assert not any(line.endswith("-") for line in layout.lines)
    assert " ".join(layout.lines).replace("  ", " ") == "NONA REIRIN!"


def test_a_long_word_may_still_be_broken(typesetter):
    region = _balloon(width=70, height=200)
    region.text_id = "Berkonvulsionalisasi"
    layout = typesetter.layout(region)
    if layout is not None:
        assert "".join(line.rstrip("-") for line in layout.lines) == region.text_id.upper()


def test_short_words_are_never_broken():
    assert not _breakable("NONA")
    assert not _breakable("--")
    assert not _breakable("TERUS--"[:4])
    assert _breakable("BERKONVULSI")


def test_every_line_fits_the_width_it_is_drawn_at(typesetter):
    """The width a line was wrapped against must be the width it is drawn at.

    Measuring at one position and drawing at another produces text that is
    silently undersized, or that overflows the balloon -- with no exception
    raised either way.
    """
    region = _balloon()
    region.text_id = "Dia nyaris tak sadarkan diri dan kadang berhalusinasi."
    layout = typesetter.layout(region)
    assert layout is not None

    mask, box = _drawable_area(region, _latin_config().typeset.padding_ratio)
    font = typesetter._font(layout.size)
    metrics = font.getMetrics()
    for index, (line, centre) in enumerate(zip(layout.lines, layout.centres)):
        baseline = layout.first_baseline + index * layout.line_height
        left = centre - font.measureText(line) / 2.0
        right = left + font.measureText(line)
        row = int(baseline - (-metrics.fAscent) / 2) - box.y1
        assert 0 <= row < mask.shape[0], "a line was placed outside the balloon"
        columns = np.flatnonzero(mask[row])
        assert columns.size, "a line was placed on an empty row"
        assert left >= box.x1 + columns[0] - 1
        assert right <= box.x1 + columns[-1] + 1


def test_lettering_is_not_grown_far_beyond_what_it_replaces(typesetter):
    """A short translation must not come back in display type."""
    region = _balloon(width=400, height=300)
    region.lines = [Box(340, 360, 460, 380), Box(340, 385, 460, 405)]
    region.text_id = "Ya."
    layout = typesetter.layout(region)
    assert layout is not None
    cfg = _latin_config().typeset
    limit = typesetter._source_size(region) * cfg.max_growth_over_source
    assert layout.size <= limit + 0.01


def test_size_follows_the_line_pitch_of_the_source(typesetter):
    """Bold lettering must come back bold.

    The size is read off the region's height over its line count. Reading it
    off the detector's line bars instead measured 10.5px on a balloon whose
    lettering was around 23px -- a faint extra bar pulled their median down --
    and the balloon came back set at two thirds the original size.
    """
    tall = _balloon(width=300, height=260)
    tall.box = Box(340, 330, 460, 470)          # 140px over 4 lines: pitch 35
    tall.lines = [Box(340, 330 + 35 * i, 460, 360 + 35 * i) for i in range(4)]
    thin = _balloon(width=300, height=260)
    thin.box = Box(340, 330, 460, 400)          # 70px over 4 lines: pitch 17.5
    thin.lines = [Box(340, 330 + 17 * i, 460, 345 + 17 * i) for i in range(4)]

    assert typesetter._source_size(tall) > typesetter._source_size(thin) * 1.8


def test_size_is_bounded_when_no_line_boxes_are_known(typesetter):
    """An empty line list must not read the whole block as one glyph."""
    region = _balloon(width=400, height=300)
    region.lines = []
    region.source_line_height = 10.0
    assert typesetter._source_size(region) < 20.0


def test_a_region_with_no_polygon_uses_its_own_box(typesetter):
    region = Region(
        id="r00",
        box=Box(100, 100, 260, 180),
        lines=[Box(100, 100, 260, 140), Box(100, 140, 260, 180)],
        status=RegionStatus.TRANSLATED,
        text_id="Diam.",
    )
    layout = typesetter.layout(region)
    assert layout is not None
    assert all(100 <= centre <= 260 for centre in layout.centres)


def test_a_slanted_balloon_does_not_collapse_the_fit(typesetter):
    """Each line is centred where it sits, not on one centre for the block.

    A shout balloon slants, so a single centre for the whole block falls
    outside some of the rows it spans. Those rows then report zero usable
    width and the layout is abandoned, which forced the text down to a size
    small enough to finish before reaching them.
    """
    text = "Penjaga bawahan, pergi selamatkan Reirin!"

    def fit(polygon: np.ndarray):
        region = Region(
            id="r00",
            box=Box(180, 130, 260, 190),
            lines=[Box(180, 130, 260, 150), Box(180, 160, 260, 180)],
            polygon=polygon,
            status=RegionStatus.TRANSLATED,
            text_id=text,
        )
        return typesetter.layout(region)

    # Same width and height, one sheared sideways with height.
    straight = fit(np.array([[100, 60], [260, 60], [260, 260], [100, 260]], np.int32))
    slanted = fit(np.array([[100, 100], [260, 60], [340, 220], [180, 260]], np.int32))

    assert straight is not None and slanted is not None
    # Centres must track the balloon rather than sit on a single column.
    assert len({round(c) for c in slanted.centres}) > 1
    # The slant itself must not cost size. Before per-line centring it did:
    # rows whose span excluded the block's one centre reported zero width.
    assert slanted.size >= straight.size * 0.75


def test_a_word_breaks_at_a_hyphen_it_already_has(typesetter):
    """"REIRIN-SAMA!!" splits where the letterer split it, adding nothing."""
    region = _balloon(width=110, height=200)
    region.lines = [Box(345, 380, 455, 400), Box(345, 405, 455, 425)]
    region.text_id = "Reirin-sama!!"
    layout = typesetter.layout(region)
    assert layout is not None
    assert layout.lines == ["REIRIN-", "SAMA!!"]
    assert not any(line.endswith("--") for line in layout.lines)


def test_colours_survive_a_render(typesetter):
    """Skia surfaces are BGRA on Windows and RGBA elsewhere.

    Reading one back as if it were always RGBA swaps red and blue, which
    turned whole reported pages solid blue.
    """
    page = np.zeros((24, 24, 3), np.uint8)
    page[:, :] = (20, 90, 200)  # BGR
    out = typesetter.render_page(page, [])
    assert tuple(int(v) for v in out[0, 0]) == (20, 90, 200)


def test_text_that_cannot_fit_is_refused_rather_than_overflowing(typesetter):
    region = Region(
        id="r00",
        box=Box(10, 10, 24, 18),
        source_line_height=4.0,
        status=RegionStatus.TRANSLATED,
        text_id="Kalimat yang jauh terlalu panjang untuk balon sekecil ini.",
    )
    assert typesetter.layout(region) is None


def test_a_refused_region_is_not_marked_typeset(typesetter):
    page = np.full((200, 200, 3), 255, np.uint8)
    region = Region(
        id="r00",
        box=Box(10, 10, 24, 18),
        source_line_height=4.0,
        status=RegionStatus.TRANSLATED,
        text_id="Kalimat yang jauh terlalu panjang untuk balon sekecil ini.",
    )
    typesetter.render_page(page, [region])
    assert region.status is RegionStatus.FAILED_FIT


# -- two blocks never letter over each other ---------------------------------


def _in_balloon(rid: str, box: Box, balloon: Box, text: str) -> Region:
    """A region whose drawable shape is the whole of one balloon.

    Which is what three of the four ways a shape is arrived at hand back: the
    balloon outline taken inside the detected box, the growth into flat
    ground, and the balloon rectangle itself all describe the balloon, not
    this region's share of it.
    """
    return Region(
        id=rid,
        box=box,
        polygon=np.array(
            [
                [balloon.x1, balloon.y1], [balloon.x2, balloon.y1],
                [balloon.x2, balloon.y2], [balloon.x1, balloon.y2],
            ],
            np.int32,
        ),
        source_line_height=13.0,
        status=RegionStatus.TRANSLATED,
        text_id=text,
    )


def _drawn_overlap(typesetter, regions, layouts) -> int:
    boxes = {
        region.id: typesetter.lines_drawn(region, layouts[region.id])
        for region in regions
        if region.id in layouts
    }
    ids = sorted(boxes)
    return sum(
        _covered(boxes[b], one)
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        for one in boxes[a]
    )


def test_two_blocks_in_one_balloon_are_not_lettered_over_each_other(typesetter):
    """The fault this was reported for: two translations, one on the other.

    Both regions are handed the whole balloon, which is what happens whenever
    the shapes are arrived at any way but the partitioned flood fill, and each
    is fitted on its own. Nothing downstream looked at the pair.
    """
    balloon = Box(100, 100, 500, 400)
    left = _in_balloon(
        "r00", Box(130, 180, 260, 320), balloon,
        "Apa yang terjadi...? Ini terlalu rumit!",
    )
    right = _in_balloon(
        "r01", Box(300, 160, 470, 340), balloon,
        "Bukankah kapten terkadang diberikan pasangan terendah?",
    )
    regions = [left, right]
    layouts = typesetter.plan(regions)
    assert _drawn_overlap(typesetter, regions, layouts) == 0


def test_the_block_reaching_into_its_neighbour_is_the_one_that_moves(typesetter):
    """Not the larger, and not the later: the one in the other's way.

    The neighbour is being drawn where its own lettering was, which is where
    it has to go; the block spilling across it has the whole rest of the
    balloon to move into.
    """
    balloon = Box(100, 100, 500, 400)
    settled = _in_balloon("r00", Box(130, 180, 260, 320), balloon, "Ini rumit!")
    spilling = _in_balloon(
        "r01", Box(300, 160, 470, 340), balloon,
        "Bukankah kapten Eagle Eye terkadang diberikan pasangan dengan "
        "peringkat yang paling rendah di antara semuanya?",
    )
    regions = [settled, spilling]
    before = typesetter.layout(settled)
    layouts = typesetter.plan(regions)
    assert _drawn_overlap(typesetter, regions, layouts) == 0
    assert layouts["r00"].centres == before.centres, "the block in place moved"


def test_a_block_with_nowhere_left_to_go_keeps_its_own_lettering(typesetter):
    """An omission is recoverable by eye; two texts over each other are not."""
    balloon = Box(100, 100, 260, 260)
    big = _in_balloon(
        "r00", Box(110, 110, 250, 250), balloon,
        "Kalimat yang cukup panjang untuk mengisi seluruh balon ini sendiri.",
    )
    second = _in_balloon("r01", Box(150, 150, 210, 210), balloon, "Ini rumit sekali!")
    regions = [big, second]
    layouts = typesetter.plan(regions)
    assert _drawn_overlap(typesetter, regions, layouts) == 0
    assert (
        RegionStatus.FAILED_FIT in {r.status for r in regions}
        or len(layouts) == 2
    )


def test_regions_that_do_not_share_a_balloon_are_fitted_as_before(typesetter):
    """The guarantee must cost nothing on the pages that were already right."""
    one = _in_balloon("r00", Box(130, 130, 260, 260), Box(100, 100, 300, 300), "Ini rumit!")
    far = _in_balloon("r01", Box(730, 130, 860, 260), Box(700, 100, 900, 300), "Ini rumit!")
    alone = typesetter.layout(one)
    layouts = typesetter.plan([one, far])
    assert layouts["r00"].centres == alone.centres
    assert layouts["r00"].size == alone.size


def test_a_block_placed_by_hand_is_not_the_one_asked_to_move(typesetter):
    """Someone looking at the page knows something the fitter does not."""
    balloon = Box(100, 100, 500, 400)
    nudged = _in_balloon("r00", Box(130, 180, 260, 320), balloon, "Ini terlalu rumit!")
    nudged.offset_x = 40.0
    automatic = _in_balloon(
        "r01", Box(300, 160, 470, 340), balloon,
        "Bukankah kapten Eagle Eye terkadang diberikan pasangan terendah?",
    )
    placed = typesetter.layout(nudged)
    layouts = typesetter.plan([nudged, automatic])
    assert _drawn_overlap(typesetter, [nudged, automatic], layouts) == 0
    assert layouts["r00"].centres == placed.centres, "the hand-placed block moved"
