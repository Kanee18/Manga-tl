"""A page the model answers badly must still come back translated, or say so.

Reported as "the text isn't being translated": six of nine balloons on a page
were left in English, and nothing in the window, the log or the summary line
said anything had gone wrong. Two faults did it, and both are here.

The model was asked for well-formed JSON and nothing more. gemma2:9b answered
the nine-line page with one line and closed the object, and on another run
wrote `"r06": {"en": ..., "id": "Seni Daois!", "r07": {...` -- one brace short,
which parsed as nothing at all and threw the whole page away. Any reply that
carried a single usable line counted as success, so the batch never asked
again for what it had not been given.

What was then left over said nothing about itself. A region the model never
mentioned keeps RegionStatus.READ, which is not a failure status, so it is not
coloured in the window, not counted in PageResult.problems and not logged.
"""

from __future__ import annotations

import json

from mangatl.config import default_config
from mangatl.pipeline.translate import (
    OllamaTranslator,
    _parse_object,
    _reply_schema,
    _unpack,
)
from mangatl.types import Box, Region, RegionStatus

# The reply gemma2:9b really sent for page 018, shortened. The brace closing
# r06 is missing, exactly as it was: every line from r06 down was lost.
BRACE_SHORT = (
    '{"r01": {"en": "LET ME GO!", "id": "Lepaskan aku!"}, '
    '"r06": {"en": "DAOIST ARTS!", "id": "Seni Daois!", '
    '"r07": {"en": "HAH..", "id": "Heh.."}}   \n\n  \n  '
)


def _page(count: int = 4) -> list[Region]:
    return [
        Region(
            id=f"r{index:02d}",
            box=Box(0, index * 40, 100, index * 40 + 30),
            text_en=f"LINE NUMBER {index} HERE",
            status=RegionStatus.READ,
        )
        for index in range(count)
    ]


def _reply(regions: list[Region]) -> str:
    return json.dumps(
        {r.id: {"en": r.text_en, "id": f"Baris nomor {r.id} di sini"} for r in regions}
    )


# -- reading a damaged reply -------------------------------------------------


def test_a_reply_one_brace_short_keeps_every_line():
    table = _parse_object(BRACE_SHORT)
    assert sorted(table) == ["r01", "r06", "r07"]
    assert _unpack(table["r06"])[0] == "Seni Daois!"
    assert _unpack(table["r07"])[0] == "Heh.."


def test_a_reply_cut_off_mid_line_keeps_the_lines_before_it():
    """The half-written one is dropped rather than lettered onto the page."""
    table = _parse_object('{"r01": {"en": "A", "id": "Satu"}, "r02": {"en": "B", "id": "Du')
    assert sorted(table) == ["r01"]


def test_a_reply_wrapped_in_an_object_is_still_read():
    table = _parse_object('{"lines": {"r01": {"en": "SILENCE.", "id": "Diam."}}}')
    assert _unpack(table["r01"]) == ("Diam.", "SILENCE.", False)


def test_prose_instead_of_a_reply_is_no_translation():
    assert _parse_object("I'm sorry, I can't help with that.") == {}


# -- asking again for what did not come back ---------------------------------


def test_the_batch_asks_again_for_the_lines_it_was_not_given():
    """A model that answers one line per request must still finish the page."""
    page = _page()
    served: list[list[str]] = []

    def one_at_a_time(self, prompt, schema=None):
        outstanding = [r for r in page if r.status is RegionStatus.READ]
        served.append([r.id for r in outstanding])
        return _reply(outstanding[:1])

    with _patched(one_at_a_time):
        OllamaTranslator(default_config()).translate_page(page, None)

    assert [r.status for r in page] == [RegionStatus.TRANSLATED] * 4
    assert served[0] == ["r00", "r01", "r02", "r03"]
    assert served[1] == ["r01", "r02", "r03"], "the settled line was asked for again"


def test_a_request_names_every_line_it_still_wants():
    page = _page()
    schemas: list[dict] = []

    def collect(self, prompt, schema=None):
        schemas.append(schema)
        return _reply(page[:1])

    with _patched(collect):
        OllamaTranslator(default_config()).translate_page(page, None)

    assert schemas[0]["required"] == ["r00", "r01", "r02", "r03"]
    assert schemas[1]["required"] == ["r01", "r02", "r03"]


def test_a_skipped_line_is_not_accused_of_leaving_english_in():
    """It was never answered at all, so that note describes nothing it did."""
    page = _page(2)
    prompts: list[str] = []

    def collect(self, prompt, schema=None):
        prompts.append(prompt)
        return _reply(page[:1])

    with _patched(collect):
        OllamaTranslator(default_config()).translate_page(page, None)

    corrective = next(p for p in prompts if "previous answer" in p)
    assert "did not include r01" in corrective
    assert "left English words" not in corrective


# -- a line that never comes back says so ------------------------------------


def test_a_line_the_model_never_returns_is_reported_as_a_failure():
    page = _page(6)

    def silence(self, prompt, schema=None):
        return "{}"

    with _patched(silence):
        OllamaTranslator(default_config()).translate_page(page, None)

    assert all(r.status is RegionStatus.FAILED_TRANSLATE for r in page)
    assert all(r.note for r in page), "a failure with no note explains nothing"
    assert not any(r.status is RegionStatus.READ for r in page)


def test_an_empty_page_asks_for_nothing():
    assert _reply_schema([]) is None


def _patched(replacement):
    from unittest.mock import patch

    return patch.object(OllamaTranslator, "_chat", replacement)
