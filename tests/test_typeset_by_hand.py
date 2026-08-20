"""Size and weight set by hand, and the search that has to honour them.

Both are adjustments rather than commands: they move what the fitter aims for,
and the fitter still refuses to put lettering outside its balloon. What these
hold down is that asking for larger type actually produces larger type -- twice
it did not, for two different reasons, and in both cases the button looked
broken rather than constrained.
"""

from __future__ import annotations

import numpy as np
import pytest

from mangatl.config import default_config
from mangatl.pipeline.typeset import Typesetter, _largest_that_fits
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
def typesetter():
    return Typesetter(_latin_config())


def _region(text: str = "Lepaskan aku!", box: Box | None = None) -> Region:
    # Deliberately short: a taller block asks for type above `max_font_size`
    # and lands on the cap at every scale, which hides whether scaling did
    # anything at all.
    box = box or Box(100, 100, 400, 180)
    return Region(
        id="r00",
        box=box,
        lines=[
            Box(box.x1, box.y1, box.x2, box.y1 + box.height // 2),
            Box(box.x1, box.y1 + box.height // 2, box.x2, box.y2),
        ],
        source_line_height=box.height / 4,
        text_id=text,
        status=RegionStatus.TRANSLATED,
        polygon=np.array(
            [
                [box.x1 - 40, box.y1 - 40],
                [box.x2 + 40, box.y1 - 40],
                [box.x2 + 40, box.y2 + 40],
                [box.x1 - 40, box.y2 + 40],
            ],
            np.int32,
        ),
    )


# -- the search --------------------------------------------------------------


def test_the_search_finds_the_largest_size_that_fits():
    """Bisection, so the answer does not depend on where the walk started."""
    found = _largest_that_fits(lambda size: "ok" if size <= 30.0 else None, 5.0, 60.0)
    assert found == "ok"


def test_the_search_gives_up_when_nothing_fits():
    assert _largest_that_fits(lambda size: None, 5.0, 60.0) is None


def test_the_search_takes_the_ceiling_when_the_ceiling_fits():
    assert _largest_that_fits(lambda size: size, 5.0, 60.0) == 60.0


# -- size --------------------------------------------------------------------


def test_asking_for_larger_type_gets_larger_type(typesetter):
    region = _region()
    plain = typesetter.layout(region)
    region.size_scale = 1.5
    bigger = typesetter.layout(region)
    assert plain is not None and bigger is not None
    assert bigger.size > plain.size


def test_asking_for_smaller_type_gets_smaller_type(typesetter):
    region = _region()
    plain = typesetter.layout(region)
    region.size_scale = 0.7
    smaller = typesetter.layout(region)
    assert smaller.size < plain.size


def test_the_global_size_cap_does_not_outrank_a_person_looking_at_the_page(typesetter):
    """`max_font_size` guards the automatic path against a nonsense measurement.

    Three of nine balloons on one page were already sitting on it, so pressing
    the enlarge button on any of them did nothing whatsoever.
    """
    cap = _latin_config().typeset.max_font_size
    region = _region("Ya!", Box(60, 60, 560, 360))
    region.source_line_height = 200.0        # would ask for far more than the cap
    at_cap = typesetter.layout(region)
    assert at_cap is not None and at_cap.size == pytest.approx(cap, abs=0.5)

    region.size_scale = 2.0
    lifted = typesetter.layout(region)
    assert lifted is not None and lifted.size > cap


def test_the_balloon_still_has_the_last_word(typesetter):
    """The one thing enlarging must never do is put text outside its shape."""
    region = _region("Kalimat yang panjang sekali untuk balon sekecil ini")
    region.polygon = np.array(
        [[100, 100], [400, 100], [400, 260], [100, 260]], np.int32
    )
    region.size_scale = 3.0
    found = typesetter.layout(region)
    assert found is not None
    width = 400 - 100
    assert all(
        found.lines and centre - width / 2 >= 90 for centre in found.centres
    ), "a line was centred outside the balloon"
    assert found.size < 3.0 * typesetter._source_size(region)


# -- weight ------------------------------------------------------------------


def test_bold_uses_the_bold_face(typesetter):
    bold = typesetter._font(30.0, bold=True).getTypeface()
    plain = typesetter._font(30.0, bold=False).getTypeface()
    assert bold.getFamilyName() != plain.getFamilyName() or bold.isBold()


def test_bold_lettering_puts_more_ink_on_the_page(typesetter):
    page = np.full((400, 500, 3), 255, np.uint8)
    region = _region()

    def inked(bold: bool) -> int:
        region.bold = bold
        region.status = RegionStatus.TRANSLATED
        drawn = typesetter.render_page(page.copy(), [region])
        return int((drawn < 128).sum())

    assert inked(True) > inked(False)


# -- sizing when the source is written the other way round -------------------


def test_a_vertical_cjk_source_is_not_sized_from_its_line_pitch():
    """`max_growth_over_source` needs both sides set the same way round.

    Japanese balloons are usually set in vertical columns, so the line head
    marks each *character* as a line -- ten of them in one 196px balloon --
    and the pitch that implies is a third of the type actually drawn. Every
    balloon on a page came out near 14pt where the source was 40.
    """
    from mangatl.pipeline.typeset import Typesetter

    cfg = _latin_config()
    latin = Typesetter(cfg)
    cfg_cjk = cfg.model_copy(deep=True)
    cfg_cjk.translate.pair = "ja-id"
    cjk = Typesetter(cfg_cjk)

    def region_from_vertical_columns() -> Region:
        # A block whose "lines" are stacked characters, in a balloon with room
        # to spare -- so what the two typesetters disagree about is the sizing
        # rule and not how much space there is.
        box = Box(100, 100, 190, 300)
        region = _region("Batas!", box)
        region.lines = [
            Box(box.x1, box.y1 + i * 20, box.x2, box.y1 + i * 20 + 18)
            for i in range(10)
        ]
        region.source_line_height = 18.0
        region.polygon = np.array(
            [[40, 60], [460, 60], [460, 340], [40, 340]], np.int32
        )
        return region

    small = latin.layout(region_from_vertical_columns())
    proper = cjk.layout(region_from_vertical_columns())
    assert small is not None and proper is not None
    assert proper.size > small.size * 1.5, (
        f"CJK sizing barely moved: {small.size:.1f} -> {proper.size:.1f}"
    )


def test_a_cjk_source_still_cannot_overflow_its_balloon():
    from mangatl.pipeline.typeset import Typesetter

    cfg = default_config().model_copy(deep=True)
    cfg.translate.pair = "ja-id"
    region = _region("Kalimat panjang sekali untuk balon kecil ini")
    region.polygon = np.array(
        [[100, 100], [400, 100], [400, 180], [100, 180]], np.int32
    )
    found = Typesetter(cfg).layout(region)
    assert found is not None
    assert found.first_baseline >= 100
