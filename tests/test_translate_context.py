"""A retried line must still be translated as part of its page.

Reported as "the translation output is incoherent". A region rejected by a
guard, or missed by the batch, was retranslated on its own -- the model got
one balloon with no idea who was speaking or what had just been said, and
answered accordingly. The page context has to survive the retry.
"""

from __future__ import annotations

from mangatl.config import default_config
from mangatl.pipeline.translate import OllamaTranslator
from mangatl.types import Box, Region, RegionStatus


def _page() -> list[Region]:
    lines = [
        ("r00", "WHAT IS IT, REIRIN?", "Ada apa, Reirin?"),
        ("r01", "YOUR HIGHNESS.", "Yang Mulia."),
        ("r02", "AREN'T YOU PUSHING YOURSELF?", ""),
        ("r03", "I APPRECIATE YOUR CONCERN.", "Terima kasih atas perhatian Anda."),
    ]
    page = []
    for index, (rid, english, indonesian) in enumerate(lines):
        page.append(
            Region(
                id=rid,
                box=Box(0, index * 40, 100, index * 40 + 30),
                text_en=english,
                text_id=indonesian,
                status=RegionStatus.READ,
            )
        )
    return page


def _translator() -> OllamaTranslator:
    # No request is made; only prompt construction is exercised.
    return OllamaTranslator(default_config())


def test_a_retry_still_sees_the_whole_page():
    page = _page()
    focus = [r for r in page if r.id == "r02"]
    with _translator() as tr:
        prompt = tr._prompt_for(page, None, focus)

    for region in page:
        assert region.text_en in prompt, f"{region.id} missing from the retry prompt"
    assert "r02" in prompt


def test_settled_lines_are_shown_with_their_translations():
    """So a retry matches the register the rest of the page already set."""
    page = _page()
    with _translator() as tr:
        prompt = tr._prompt_for(page, None, [p for p in page if p.id == "r02"])
    assert "[done: Ada apa, Reirin?]" in prompt
    assert "[done: Yang Mulia.]" in prompt


def test_the_model_is_asked_for_only_the_focused_ids():
    page = _page()
    with _translator() as tr:
        prompt = tr._prompt_for(page, None, [p for p in page if p.id == "r02"])
    assert "Translate only these ids: r02" in prompt
    assert "do not return them" in prompt


def test_a_full_page_request_has_no_focus_wording():
    page = _page()
    with _translator() as tr:
        prompt = tr._prompt_for(page, None, page)
    assert "Translate only these ids" not in prompt
    assert "belong to one page" in prompt


def test_settled_lines_are_not_marked_done_in_a_full_request():
    page = _page()
    with _translator() as tr:
        prompt = tr._prompt_for(page, None, None)
    assert "[done:" not in prompt


def test_one_lines_translation_cannot_be_given_to_another():
    """The model sometimes attaches an answer to the wrong id.

    A caption reading "under the guise of Etiquette Training" came back with
    the translation belonging to a balloon about jewels, and it was lettered
    into both. The text is fluent Indonesian, so only the collision reveals it.
    """
    page = _page()
    settled = next(r for r in page if r.id == "r00")
    settled.status = RegionStatus.TRANSLATED

    victim = next(r for r in page if r.id == "r02")
    with _translator() as tr:
        assert not tr._accept(victim, settled.text_id, page=page)
    assert victim.status is RegionStatus.FAILED_TRANSLATE
    assert "r00" in victim.note


def test_identical_source_lines_may_share_a_translation():
    """Two balloons both reading "REIRIN!" should translate the same way."""
    page = _page()
    first, second = page[0], page[1]
    second.text_en = first.text_en
    first.status = RegionStatus.TRANSLATED

    with _translator() as tr:
        assert tr._accept(second, first.text_id, page=page)
    assert second.text_id == first.text_id
