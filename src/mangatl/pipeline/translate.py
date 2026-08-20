"""B1 -- translate a page's lettering from English to Indonesian, via Ollama.

A whole page goes in one request, in reading order. That is not a batching
optimisation: it is what lets the model see that the balloon reading "Explain."
is answered by the three that follow it, and keep one speaker's register
across all of them. Translating balloon by balloon produces text that is
correct sentence by sentence and inconsistent down the page -- the register
slips between formal and casual, and a name comes back spelled two ways.

Three things are checked on the way back, because a translator that fails
quietly is worse than one that fails loudly:

* every requested id came back, and nothing else did;
* the result is not the English that was sent (a model that gives up tends to
  echo, and echoed English typeset in Anime Ace looks deliberate);
* the result is not obviously still in the source language.

Anything that fails lands on `RegionStatus.FAILED_TRANSLATE`, and since
inpainting runs after this stage, such a region simply keeps its original
pixels rather than being erased and left blank.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx

from ..config import Config
from ..languages import (
    CJK,
    ENGLISH,
    LATIN_WORD,
    Pair,
    is_sound_effect,
    is_stretched_noise,
    pair_for,
)
from ..types import Region, RegionStatus


class TranslationError(RuntimeError):
    """Ollama could not be reached, or refused the request."""


# English function words. None is a valid Indonesian word, and none can be
# mistaken for a name, so a single one is proof that part of the source came
# back untranslated. Requiring two was too lax -- "Istana Selatan Terttama
# House of Sn" carries exactly one ("of") and shipped half-English to a page.
#
# Nouns are deliberately absent, including the ones that keep turning up
# untranslated: empress, emperor, maiden, house, court, territory, highness.
# Listing them made things worse rather than better. A correct translation
# reading "...menjadi Empress berikutnya" was rejected, the corrective retry
# told the model the word "empress" was unacceptable, and it complied by
# inventing "EMPERIUT" -- a non-word where there had been one English one.
# Every model tried did this: qwen produced "EMPERESS" and "EMPRESI" the same
# way. A rejection loop can only push a model off a word it wants; it cannot
# teach it the right one. That is what the glossary is for, and glossary terms
# are injected into the prompt where they steer the model instead of fighting it.
_ENGLISH_MARKERS = frozenset(
    {
        "the", "of", "and", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "this", "that", "these", "those", "with", "from",
        "they", "them", "their", "she", "her", "hers", "his", "him", "he",
        "you", "your", "yours", "would", "could", "should", "shall",
        "what", "when", "where", "which", "who", "whom", "whose", "why",
        "however", "although", "because", "there", "here", "then", "than",
    }
)

_WORD = LATIN_WORD

# The keys inside one entry, as opposed to the ids that name entries.
_ENTRY_KEYS = frozenset({"en", "english", "id", "indonesian", "target"})

def system_prompt(pair: Pair) -> str:
    """What the model is told, for one language pair.

    Assembled rather than written out per pair, because the shape is the same
    every time -- repair the reading, then translate it -- and only the two
    ends differ. Five hand-written copies would mean five copies of the JSON
    contract to keep in step, and the contract is the part that must not drift.
    """
    source, target = pair.source, pair.target
    damage = source.ocr_damage or (
        "Letters are misread as similar-looking ones, whole words are dropped, "
        "and final punctuation is lost."
    )
    vertical = (
        "\n\nLettering may have been set vertically. It is given to you already "
        "in reading order, so treat each line as it stands."
        if source.is_cjk
        else ""
    )
    shape = json.dumps(
        {
            "r01": {
                "en": source.example_silence,
                "id": target.example_silence,
                "sfx": False,
            }
        },
        ensure_ascii=False,
    )
    return (
        f"You are a professional manga scanlator producing the {target.name} "
        f"edition of a {source.name} manga. You are given the lettering of one "
        "page, in reading order.\n\n"
        "Handle every line in two steps.\n\n"
        "STEP 1 -- repair. The lines come from OCR of comic lettering and are "
        "damaged. Using the neighbouring lines as context, write down the "
        f"{source.name} sentence the letterer actually wrote. Repair it fully "
        "-- a half-repaired sentence produces a mistranslation.\n\n"
        f"{damage}\n\n"
        "Dropping a negation reverses what the line means, and a reversed line "
        "is far worse than a clumsy one -- it makes the page contradict its own "
        "story. If a line is too damaged to recover, translate what is actually "
        "written as closely as you can rather than inventing a different "
        f"sentence that reads well.{vertical}\n\n"
        f"STEP 2 -- translate that repaired {source.name} into {target.name}.\n\n"
        f"{target.name} requirements:\n"
        f"{target.style}\n"
        "- Keep it SHORT. Each line is typeset back into the balloon it came "
        "from, so prefer the most concise natural wording.\n"
        "- Preserve punctuation exactly: ellipses (...), double dashes (--), "
        "and exclamation and question marks.\n"
        "- No notes, no explanations, no romanisation.\n\n"
        "Sound effects. Some lines are not speech at all but noises drawn "
        "into the artwork -- CRACK, THUD, SQUEEZE, RUSTLE, a footstep. Set "
        '"sfx" to true for those, and leave "id" as the line itself. They '
        "are part of the drawing rather than part of the script, and "
        "replacing them would rub out what the artist drew. Set "
        '"sfx" to false for anything a character says, thinks or '
        "narrates, however short -- a one-word reply is still speech.\n\n"
        f"Worked examples:\n{target.examples or source.examples}\n\n"
        "Reply with a JSON object only. Map each given id to an object with two "
        f'keys, "en" for the repaired {source.name}, "id" for the '
        f'{target.name}, and "sfx" for whether the line is a drawn sound '
        "effect. Include every id you were given and no others. For "
        f"example:\n{shape}"
    )


@dataclass
class TranslationMemory:
    """Recent source/target pairs, carried across the pages of a chapter.

    A chapter is one conversation between the same characters, so the way a
    name or a form of address was rendered on page 3 is the right way to
    render it on page 4. Without this each page is translated by a model with
    no memory of the last one, and the reader watches a character's name and
    honorific change between pages.
    """

    limit: int = 24
    pairs: list[tuple[str, str]] = field(default_factory=list)

    def remember(self, source: str, target: str) -> None:
        if source and target:
            self.pairs.append((source, target))
            del self.pairs[: max(0, len(self.pairs) - self.limit)]

    def as_prompt(self) -> str:
        if not self.pairs:
            return ""
        lines = "\n".join(f"{en} -> {id_}" for en, id_ in self.pairs[-self.limit :])
        return (
            "Earlier lines from this same chapter, for consistency of names, "
            "forms of address and register:\n" + lines + "\n\n"
        )


def leftover_source(target: str, pair: Pair) -> set[str]:
    """Source-language fragments still present in a supposed translation.

    Two different tests, because the two situations are not alike.

    From a CJK source into a Latin target, any kana or han character in the
    answer is untranslated source, full stop. There is no ambiguity to weigh
    and nothing to tune -- English and Indonesian do not contain those
    characters at all.

    From English into Indonesian the test has to be a word list, and which
    words are on it matters: see `_ENGLISH_MARKERS`. Nothing like it exists
    for the reverse direction, which is why translating *into* English cannot
    use this test and does not need to.
    """
    if pair.source.is_cjk and not pair.target.is_cjk:
        return set(CJK.findall(target))
    if pair.source is ENGLISH and pair.target is not ENGLISH:
        return set(_WORD.findall(target.strip().lower())) & _ENGLISH_MARKERS
    return set()


def looks_untranslated(source: str, target: str, pair: Pair) -> bool:
    """Whether a returned line is really still the source, whole or in part."""
    a, b = source.strip().lower(), target.strip().lower()
    if not b:
        return True
    # Echoing the source back is the failure a model falls into when it gives
    # up. Counted in words for a spaced script and in characters for one
    # without, since "私は" is a whole sentence and splits into one token.
    #
    # A line with none of the source script in it cannot have been echoed in
    # the sense that matters: it is a name, a place or a shout that is written
    # the same way in both languages, and returning it unchanged is the
    # correct answer. "ABYDOS" and "SERANG BANK" were both rejected as echoes
    # on the reported page and shipped untranslated, which is precisely the
    # outcome the guard exists to prevent.
    if a == b and not (pair.source.is_cjk and not CJK.search(a)):
        units = len(a) if pair.source.is_cjk else len(_WORD.findall(a))
        if units >= (4 if pair.source.is_cjk else 3):
            return True
    return bool(leftover_source(target, pair))


class OllamaTranslator:
    """A translation backend backed by a local Ollama server."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._t = cfg.translate
        self._pair = pair_for(self._t.pair)
        self._prompt = system_prompt(self._pair)
        self._client = httpx.Client(
            base_url=self._t.host.rstrip("/"), timeout=self._t.timeout
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaTranslator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- server ---------------------------------------------------------

    def health(self) -> tuple[bool, str]:
        """Whether Ollama is up and the configured model is present."""
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return False, (
                f"cannot reach Ollama at {self._t.host} ({exc.__class__.__name__}). "
                "Start it with `ollama serve`."
            )
        names = {m.get("name", "") for m in response.json().get("models", [])}
        if self._t.model in names or f"{self._t.model}:latest" in names:
            return True, f"{self._t.model} ready"
        return False, (
            f"model {self._t.model!r} is not installed. "
            f"Pull it with `ollama pull {self._t.model}`, or choose one of: "
            + ", ".join(sorted(names)[:6])
        )

    def list_models(self) -> list[str]:
        """Models installed on the server, or an empty list if it is down."""
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        names = [str(m.get("name", "")) for m in response.json().get("models", [])]
        return sorted(name for name in names if name)

    def _chat(self, prompt: str, schema: dict[str, object] | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._t.model,
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # A schema naming every id, rather than a bare "json". Asked only
            # for well-formed JSON, a 9B model answers a nine-line page with
            # one line and closes the object, or drops a brace halfway down
            # and makes the whole reply unparseable. Both were measured on
            # gemma2:9b against page 018 of a chapter: "json" returned the
            # whole page once in three runs, the schema three times in three.
            # Decoding is constrained to the grammar, so the reply cannot end
            # early and cannot be missing a brace.
            "format": schema if schema is not None else "json",
            "options": {
                "temperature": self._t.temperature,
                "top_p": self._t.top_p,
                "num_ctx": self._t.num_ctx,
                # Fixed by default, so re-running a chapter gives the same
                # pages. Without it the same balloon comes back worded
                # differently each run -- one wording fits its balloon and the
                # next does not, and a page that was fine yesterday has a
                # region in English today for no reason anyone can see.
                **({} if self._t.seed is None else {"seed": self._t.seed}),
            },
        }
        # Reasoning models interleave their scratchpad with the answer unless
        # told not to. The field is rejected by models that cannot think, so
        # it is only sent to the ones that can.
        if "qwen3" in self._t.model or "deepseek-r1" in self._t.model:
            payload["think"] = False

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Ollama before 0.5 has no structured outputs and rejects a schema
            # outright. Falling back keeps those servers working.
            if schema is not None and exc.response.status_code == 400:
                return self._chat(prompt)
            raise TranslationError(f"Ollama request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TranslationError(f"Ollama request failed: {exc}") from exc
        return str(response.json().get("message", {}).get("content", ""))

    # -- translation ----------------------------------------------------

    def translate_page(
        self, regions: list[Region], memory: TranslationMemory | None = None
    ) -> None:
        """Translate every readable region on a page, in place."""
        self._mark_sound_effects(regions)
        pending = [r for r in regions if r.status is RegionStatus.READ and r.text_en]
        if not pending:
            return

        if self._t.page_batch:
            done = self._translate_batch(pending, pending, memory)
            missing = [r for r in pending if r.id not in done]
            # One corrective pass for the whole page, not one per region. The
            # obvious repair -- retry each rejected region with feedback --
            # costs a request per region per attempt, and turned a page that
            # took 20 seconds into one that took 507.
            if missing:
                done = self._translate_batch(
                    pending, missing, memory, corrective=True
                )
                missing = [r for r in missing if r.id not in done]
        else:
            missing = pending

        # Every request now carries the whole page, so a per-region retry
        # costs about what a whole-page one does. Running them for a long tail
        # of stubborn lines is how a page reached 445 seconds. The corrective
        # pass has already shown the model the page and named its mistake; a
        # third attempt on the same line rarely lands, so only a short tail is
        # worth paying for and the rest keep their English.
        for region in missing[: self._t.max_single_retries]:
            self._translate_one(pending, region, memory)

        # A line the model simply never mentioned is the one failure the rest
        # of the program cannot see: it keeps RegionStatus.READ, which is not
        # a failure, so it is not coloured in the window, not counted in
        # PageResult.problems and not written to the log -- the page ships
        # with an English balloon and nothing anywhere says so. Whatever is
        # still unanswered by this point has been through the batch, the
        # corrective pass and, for up to `max_single_retries` of it, a
        # request of its own.
        for region in pending:
            if region.status is RegionStatus.READ:
                region.fail(
                    RegionStatus.FAILED_TRANSLATE,
                    "the model never returned this line",
                )

        if memory is not None:
            for region in pending:
                if region.status is RegionStatus.TRANSLATED:
                    memory.remember(region.text_en, region.text_id)

    def _prompt_for(
        self,
        page: list[Region],
        memory: TranslationMemory | None,
        focus: list[Region] | None = None,
    ) -> str:
        """The request for a page, optionally asking for only some of it.

        The whole page is always listed, even when only a few lines are
        wanted. Retrying just the rejected lines seems obviously right and is
        the reason retried balloons came back incoherent: the model was handed
        three fragments with no idea who was speaking or what had been said,
        and it answered accordingly. Lines already settled are shown with
        their agreed translation, so a retry matches the register of the page
        it belongs to instead of inventing a new one.
        """
        parts: list[str] = []
        if memory is not None:
            parts.append(memory.as_prompt())
        if self._t.glossary:
            terms = "\n".join(f"{k} -> {v}" for k, v in self._t.glossary.items())
            parts.append(f"Always render these terms this way:\n{terms}\n\n")

        wanted = {r.id for r in focus} if focus is not None else {r.id for r in page}
        listing = "\n".join(
            f"{r.id}: {r.text_en}"
            + ("" if r.id in wanted or not r.text_id else f"   [done: {r.text_id}]")
            for r in page
        )
        if focus is None or len(wanted) == len(page):
            instruction = (
                f"Translate each line into {self._pair.target.name}. The lines "
                "are in reading order and belong to one page."
            )
        else:
            instruction = (
                "Below is a whole page of lettering in reading order. Lines "
                "marked [done: ...] are already translated and are given for "
                "context only -- do not return them. Translate only these "
                f"ids: {', '.join(sorted(wanted))}. Return only those ids."
            )
        parts.append(f"{instruction}\n\n{listing}")
        return "".join(parts)

    def _translate_batch(
        self,
        page: list[Region],
        focus: list[Region],
        memory: TranslationMemory | None,
        corrective: bool = False,
    ) -> set[str]:
        """Requests for a page. Returns the ids that came back usable.

        A retry asks only for the lines still outstanding, and keeps whatever
        earlier attempts settled. Returning on the first reply that carried
        anything at all is what let most of a page go untranslated in
        silence: a nine-line page came back holding one line, that counted as
        success, and the eight the model had quietly dropped were left to a
        single corrective pass and then to nothing.
        """
        outstanding = {r.id: r for r in focus}
        done: set[str] = set()
        attempts = 1 if corrective else self._t.max_retries

        for attempt in range(attempts):
            wanted = [r for r in focus if r.id in outstanding]
            prompt = self._prompt_for(page, memory, wanted)
            if corrective:
                prompt = self._correction(wanted) + prompt

            try:
                raw = self._chat(prompt, _reply_schema(r.id for r in wanted))
            except TranslationError:
                # A corrective pass is a bonus, so a dead server there is not
                # worth failing the page over. The first pass is the page,
                # and if it never landed a line there is nothing to go on.
                if not corrective and attempt == attempts - 1 and not done:
                    raise
                continue

            for key, value in _parse_object(raw).items():
                region = outstanding.get(str(key).strip())
                if region is None:
                    continue
                target, repaired, sfx = _unpack(value)
                if target and self._accept(region, target, repaired, page, sfx):
                    done.add(region.id)
                    del outstanding[region.id]
            if not outstanding:
                break
        return done

    def _correction(self, focus: list[Region]) -> str:
        """What to tell the model its last answer got wrong.

        Lines it mangled and lines it never mentioned are different mistakes
        and cannot share a note: telling a model it left English in a line it
        never answered describes a fault it did not commit, and gives it
        nothing to act on.
        """
        stray = sorted(
            {w for r in focus for w in leftover_source(r.rejected, self._pair)}
        )
        skipped = [r.id for r in focus if not r.rejected]
        notes: list[str] = []
        if stray:
            notes.append(
                f"Your previous answer left {self._pair.source.name} in "
                f"({', '.join(stray)}). Translate every word into "
                f"{self._pair.target.name}. Keep only proper names as they "
                "are written."
            )
        if skipped:
            notes.append(
                "Your previous answer did not include "
                + ", ".join(skipped)
                + ". Every id you are asked for must appear in the reply."
            )
        return " ".join(notes) + "\n\n" if notes else ""

    def _translate_one(
        self,
        page: list[Region],
        region: Region,
        memory: TranslationMemory | None,
    ) -> None:
        """Fall back to a request for a single region.

        Each retry is told what was wrong with the last answer rather than
        repeating the same request. A model that left English in will leave it
        in again if asked the same way, and naming the offending words is
        usually enough to get a complete translation on the second attempt.
        """
        base = self._prompt_for(page, memory, [region])
        prompt = base
        # One attempt. This runs only for lines the page batch and the
        # corrective pass have both already failed on.
        for _ in range(1):
            try:
                raw = self._chat(prompt, _reply_schema([region.id]))
            except TranslationError as exc:
                region.fail(RegionStatus.FAILED_TRANSLATE, str(exc))
                return
            table = _parse_object(raw)
            value = table.get(region.id) if table else None
            if value is None and table and len(table) == 1:
                value = next(iter(table.values()))  # right answer, wrong key
            target, repaired, sfx = _unpack(value)
            if target and self._accept(region, target, repaired, page, sfx):
                return
            stray = leftover_source(target, self._pair) if target else set()
            if stray:
                prompt = (
                    f"{base}\n\nYour previous answer was "
                    f"{target!r}. It still contains {self._pair.source.name}: "
                    f"{', '.join(sorted(stray))}. Translate every word into "
                    f"{self._pair.target.name}. Keep only proper names as they are."
                )
            else:
                prompt = base
        region.fail(RegionStatus.FAILED_TRANSLATE, "no usable translation returned")

    def _mark_sound_effects(self, regions: list[Region]) -> None:
        """Settle the noises before the model is asked about anything.

        The reply already carries an "sfx" flag and the prompt names the case,
        so this is not the mechanism -- it is the floor under it. Whether a
        line is a drawn noise is a judgement, and a small model gets the loud
        ones right and the quiet ones wrong: "SQUEEZE" lettered over a clasped
        hand came back as speech, which means the drawing is erased and the
        word set again in the comic face. Asking harder does not fix that, and
        the words it happens on are a short, closed list -- so the list
        decides, and the model is only asked about the rest.

        The list can only recognise a word it is given, and the words it most
        needs to recognise are the ones OCR is worst at, because they are
        drawn by hand rather than set: "GUUUURRRGLE", stretched across a
        panel, was read as "GUUUURRAGIE AND" and neither the list nor the
        model knew it. So `is_stretched_noise` asks the other question -- not
        which word this is, but whether it is spelled the way a letterer
        spells a noise -- and that survives the damage. See `_is_drawn_noise`
        for why it is only asked of some regions.

        Settling it here rather than in `_accept` also takes those lines out
        of the request entirely: a line nobody asks about cannot be answered
        wrongly, and a page of noises costs nothing to translate.
        """
        if self._t.translate_sfx:
            return
        for region in regions:
            if region.status is not RegionStatus.READ or not region.text_en:
                continue
            if self._is_drawn_noise(region, region.text_en):
                region.text_en_fixed = " ".join(region.text_en.split()).strip()
                region.fail(RegionStatus.SKIPPED_SFX, "a drawn sound effect")

    def _is_drawn_noise(self, region: Region, text: str) -> bool:
        """Whether `text` is a noise the artist drew rather than a line.

        Two tests, and the second is asked only of lettering nothing on the
        page encloses. A stretched spelling is how a letterer draws a noise
        out to fit the picture, but it is also how somebody shouts: over three
        chapters, every reading with a stretched run that was *in a balloon*
        turned out to be speech -- "NOOOO! YOU DON'T UNDERSTAND!", "AAHHH!!",
        "Pfff... ha ha ha." -- and every one drawn onto bare artwork turned
        out to be a noise. The balloon is what tells them apart, so it is what
        the test is gated on.

        Deliberately not gated on size as well. Drawn lettering is large, and
        measuring how large is easy, but it decides nothing: across the same
        three chapters, the lettering set at three to five times the page's
        dialogue was half noises the artist drew and half characters
        shouting -- "Nope", "MAYBE THREE...", "HOW COULD THIS BE?!" are all
        set that way and all speech. There is no threshold between them, and
        a wrong answer here leaves a line of real dialogue in English.
        """
        if is_sound_effect(text, self._pair.source):
            return True
        return region.is_unbounded and is_stretched_noise(text, self._pair.source)

    def _accept(
        self,
        region: Region,
        candidate: str,
        repaired: str = "",
        page: list[Region] | None = None,
        sfx: bool = False,
    ) -> bool:
        # Both tests get a second look here, at the repaired reading. Before
        # the request they only saw what OCR returned, and OCR returns
        # "5QUEEZE" for a word drawn by hand often enough to matter; repairing
        # it is the one thing the model is asked to do first, so by now it
        # says SQUEEZE.
        if repaired and self._is_drawn_noise(region, repaired):
            sfx = True
        if sfx and not self._t.translate_sfx:
            # Lettering drawn into the artwork rather than set into a balloon.
            # Erasing it would take a piece of the drawing with it, and the
            # replacement would be the comic face where the artist drew by
            # hand, so it keeps its own pixels.
            region.text_en_fixed = " ".join(repaired.split()).strip()
            region.fail(RegionStatus.SKIPPED_SFX, "a drawn sound effect")
            return True

        text = " ".join(candidate.split()).strip()
        if not text:
            region.fail(RegionStatus.FAILED_TRANSLATE, "empty translation")
            return False
        if looks_untranslated(region.text_en, text, self._pair):
            region.rejected = text
            region.fail(
                RegionStatus.FAILED_TRANSLATE,
                f"still {self._pair.source.name}: {text!r}",
            )
            return False

        # Two different lines cannot have the same translation. When a whole
        # page is translated in one request the model addresses each line by
        # id, and it sometimes attaches one line's answer to another's id: a
        # caption reading "under the guise of Etiquette Training" came back
        # with the translation belonging to a balloon about jewels and grace,
        # which was then lettered into both balloons. Nothing else catches
        # this -- the text is fluent Indonesian and passes every other check.
        if page is not None:
            key = text.casefold()
            twin = next(
                (
                    other
                    for other in page
                    if other.id != region.id
                    and other.status is RegionStatus.TRANSLATED
                    and other.text_id.casefold() == key
                    and other.text_en.casefold() != region.text_en.casefold()
                ),
                None,
            )
            if twin is not None:
                region.rejected = text
                region.fail(
                    RegionStatus.FAILED_TRANSLATE,
                    f"same translation as {twin.id}, which came from different text",
                )
                return False
        region.text_id = text
        region.text_en_fixed = " ".join(repaired.split()).strip()
        region.status = RegionStatus.TRANSLATED
        region.note = ""
        return True


def _unpack(value: object) -> tuple[str, str, bool]:
    """(target, repaired source, is a sound effect) from one entry of a reply.

    The three-field form is what is asked for, but a plain string is accepted
    too: a smaller model that drops the structure under load is still giving a
    usable translation, and rejecting the page over its shape would throw that
    away. A reply in that shape simply says nothing about sound effects.
    """
    if isinstance(value, str):
        return value, "", False
    if isinstance(value, dict):
        target = value.get("id") or value.get("indonesian") or value.get("target")
        repaired = value.get("en") or value.get("english") or ""
        sfx = value.get("sfx")
        if isinstance(target, str):
            return (
                target,
                repaired if isinstance(repaired, str) else "",
                sfx is True or str(sfx).strip().lower() == "true",
            )
    return "", "", False


def _reply_schema(ids: Iterable[str]) -> dict[str, object] | None:
    """The shape a reply has to have: an entry for every id, and no less.

    Ollama constrains decoding to this, which is the difference between a
    model that answers a nine-line page and one that answers the first line
    and closes the object. It also makes a reply that cannot be parsed at all
    impossible, which is worth as much: a single dropped brace used to cost
    the whole page rather than the one line it damaged.
    """
    wanted = [str(i) for i in ids]
    if not wanted:
        return None
    entry = {
        "type": "object",
        "properties": {
            "en": {"type": "string"},
            "id": {"type": "string"},
            # Asked for as part of the reply rather than guessed afterwards.
            # Whether a line is a drawn sound effect is a question about what
            # the words *are* -- "CRACK" is one and "MOTHER..." is not -- and
            # no measurement of the box answers it. Measured over four pages,
            # real sound effects came out *smaller* than real captions, so the
            # size rule this once had could never have worked.
            "sfx": {"type": "boolean"},
        },
        "required": ["en", "id", "sfx"],
    }
    return {
        "type": "object",
        "properties": {i: dict(entry) for i in wanted},
        "required": wanted,
    }


def _parse_object(raw: str) -> dict[str, object]:
    """The id -> entry table in a reply, tolerating fencing and damage."""
    text = raw.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {}
    body = text[start:]
    end = body.rfind("}")
    for candidate in (text, body[: end + 1] if end > 0 else "", _repair(body)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _flatten(parsed)
    return {}


def _repair(body: str) -> str:
    """A damaged object cut back to its last whole entry and closed.

    Two kinds of damage are worth surviving, because both cost a whole page
    otherwise. A reply cut off mid-entry leaves every earlier entry intact and
    unreachable. A reply missing one closing brace -- gemma2 wrote
    `"r06": {"en": ..., "id": "Seni Daois!", "r07": {...` on a page and every
    line from r06 down was thrown away -- is whole apart from that brace, and
    nests the rest of the page inside the entry it damaged, where `_flatten`
    finds it again.

    The cut is taken at the last point the object was one level deep with an
    entry just closed, so a half-written translation is dropped rather than
    lettered onto a page.
    """
    depth = 0
    in_string = escaped = False
    cut = -1
    for index, char in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 1:
                cut = index
    return body[: cut + 1] + "}" if cut > 0 else ""


def _flatten(table: dict[str, object]) -> dict[str, object]:
    """Every id -> entry pair in a reply, however deeply it sits.

    A reply repaired by `_repair` has the tail of the page nested inside the
    entry whose brace went missing, and a model asked for a flat object
    sometimes wraps one round it anyway. Neither is a reason to lose a line:
    ids the caller did not ask about are dropped by `_translate_batch`, so
    reaching too far here costs nothing.
    """
    found: dict[str, object] = {}

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                if _unpack(value)[0]:
                    found.setdefault(str(key), value)
                walk(value)
            elif isinstance(value, str) and str(key).lower() not in _ENTRY_KEYS:
                found.setdefault(str(key), value)

    walk(table)
    return found
