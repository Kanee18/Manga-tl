"""The language pairs this tool translates, and what each one changes.

A pair is not just two names in a prompt. It decides four things that are
wrong if they are left at the English-to-Indonesian defaults:

* **How the source is repaired.** OCR damage is script-specific. Latin
  lettering comes back with `1` for `I` and `5` for `S`; Japanese comes back
  with a wrong but visually similar kanji, or with small kana read as full
  size. Telling the model to expect the wrong kind of damage wastes the step.

* **How a line joins up.** Latin text breaks across lines with spaces between
  the pieces and sometimes a hyphen to heal. Japanese and Chinese have no
  spaces, so joining their lines with one inserts a gap that was never there.

* **What proves a line came back untranslated.** For an English source the
  test is a list of English function words, because Indonesian shares no
  short words with English. That test is useless when the *target* is English
  -- it would reject every correct translation. With a CJK source and a Latin
  target the test is far simpler and stronger: any kana or han character in
  the answer is source text that never got translated.

* **How the target should read.** Indonesian needs its kami/kita distinction
  and its rendering of ranks; English manga lettering has its own conventions
  and needs none of that.

Adding a pair means adding a `Language` and listing the combination in
`PAIRS`. Nothing else in the pipeline knows a language name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Kana, han, and the full-width forms that come with them. A Latin target
# containing any of these is carrying source text that was never translated.
CJK = re.compile(
    r"[぀-ゟ"      # hiragana
    r"゠-ヿ"       # katakana
    r"㐀-䶿"       # han, extension A
    r"一-鿿"       # han
    r"豈-﫿"       # han, compatibility
    r"ｦ-ﾟ]"      # half-width katakana
)

# Word characters in a Latin script, for the English-marker test.
LATIN_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class Language:
    """One language, and what the pipeline needs to know about it."""

    code: str
    name: str
    # Whether words are separated by spaces. False for Japanese and Chinese,
    # which decides how recognised lines are joined back together.
    spaced: bool = True
    # What OCR gets wrong in this script, told to the model as the repair step.
    ocr_damage: str = ""
    # How the target should read, when this language is the target.
    style: str = ""
    # Worked examples, in this language, when it is the target.
    examples: str = ""
    # One short line, used in the reply-shape example so the model is shown
    # the contract in the scripts it will actually be working in.
    example_silence: str = "SILENCE."
    # Noises this language's letterers draw into the artwork. See
    # `is_sound_effect` for what a line has to look like to count as one.
    sfx_words: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_cjk(self) -> bool:
        return not self.spaced


# Noises, as scanlators letter them into English pages. The model is asked
# whether a line is a drawn sound effect and is right about the loud ones, but
# a 2B model is not reliably right about the quiet ones: "SQUEEZE" over a
# clasped hand came back as speech and shipped translated, which erases what
# the artist drew and sets a word in the comic face where the lettering was
# hand-drawn.
#
# What is on this list is the constraint that matters, because a false
# positive leaves a line of real dialogue in English. So it holds words that
# are a noise or a drawn action and are never a whole line of dialogue by
# themselves. Interjections are deliberately absent -- "UGH", "HUH", "TSK" and
# "AHEM" are things a character says, and a one-word reply is still speech.
# So are verbs a character could plausibly shout on their own: STOP, MOVE,
# RUN, GRAB.
ENGLISH_SFX = frozenset(
    """
    bam bang boom bonk bump clang clank clap clatter click clink clunk crack
    crash crunch ding dong donk gong kaboom kablam klang knock
    pow slam smack smash snap tap thud thump thunk thwack tonk wham whack
    whomp whump plop plunk pomf poof pop puff

    fwip fwoosh swish swoosh whoosh whish whir whirr vroom zoom zip whizz whiz
    flap flutter rustle rumble roar hiss sizzle crackle fizz splash splish
    splosh sploosh slosh drip dribble plip plink gurgle glug squelch

    squeeze squish clench clutch grip hug pat rub poke prod nudge twitch
    tremble shiver shudder quiver wobble sway jolt jerk flinch gulp gasp sigh
    blush throb pound fidget stumble tumble wince slump flop tug yank swipe
    slash slice stab thwip snip scritch scratch scrape screech squeak creak
    crinkle crumple rip shove

    badump ba-dump ba-thump doki dokun gogogo zawa shaa
    """.split()
)


ENGLISH = Language(
    code="en",
    name="English",
    sfx_words=ENGLISH_SFX,
    ocr_damage="""\
The confusions to expect are between characters that look alike: Z and D, 1 \
and I or l, 0 and O, 5 and S, 7 and T or I, 6 and G, 8 and B, 4 and A, rn and \
m, C and G. Repair the word to the nearest real English word, keeping every \
prefix: "ZISAPPEAR" is "DISAPPEAR", not "APPEAR".

A digit standing where a word belongs is a misread letter, not a number. \
Comic dialogue rarely contains figures, so read them back as letters: \
"REPEATED 7 AFTER HER" is "REPEATED IT AFTER HER", and "5HE WA5 GONE" is \
"SHE WAS GONE". Leave a digit alone only when the line is genuinely about a \
quantity, a date or an age.""",
    style="""\
- Natural, idiomatic English of the kind found in published manga in English.
- Match tone: formal stays formal, a shout stays short and forceful.
- Translate idioms as idioms, never word for word.
- Japanese honorifics are dropped where English would not use them, or \
rendered as a title ("Ojou-sama" -> "my lady"). Never leave -san, -sama, \
-chan or -kun attached unless the series is known for keeping them.
- Keep names in the order and spelling the source uses.
- Chapter headings are translated too.""",
    examples="""\
  "オレはもう限界だ"
      source: オレはもう限界だ
      English: I'm at my limit.
  "何を言っているの？"
      source: 何を言っているの？
      English: What are you talking about?""",
)

INDONESIAN = Language(
    code="id",
    name="Indonesian",
    example_silence="Diam.",
    style="""\
- Natural, idiomatic Indonesian of the kind found in published manga. Never \
word-for-word from the source.
- Every word must be Indonesian. Do not leave source words in ("doctor" -> \
"dokter"), and never invent a word by giving a foreign one an Indonesian \
ending.
- Titles and ranks are translated, not carried over. The usual renderings are \
emperor -> kaisar, empress -> permaisuri, crown princess -> putri mahkota, \
your highness / your majesty -> yang mulia, house (a noble family) -> \
keluarga, territory -> wilayah, palace -> istana, court -> istana. If a rank \
has no settled Indonesian form, choose one that fits the register and use the \
same one every time -- never invent a word by respelling the foreign one.
- "Lady <Name>" is "Nona <Name>". "Our lady" is "Nona kami". Keep names \
spelled as in the source.
- "we/our" excluding the listener is "kami", including them "kita".
- Match tone: formal stays formal, a shout stays short and forceful.
- Translate idioms as idioms. "At this rate" is about a situation continuing, \
not about speed. "It can't be helped" is "Ya sudah, mau bagaimana lagi".
- Chapter headings are translated too: "Chapter" is "Bab", "Part" is "Bagian".""",
    examples="""\
  "1 CAN'T BEL1EVE 1T."
      repaired: I CAN'T BELIEVE IT.
      Indonesian: Aku tak percaya.
  "AT THIS RATE--"
      repaired: AT THIS RATE--
      Indonesian: Kalau begini terus--
  "PLEASE FORG1VE ME, YOUR H1GHNESS."
      repaired: PLEASE FORGIVE ME, YOUR HIGHNESS.
      Indonesian: Mohon ampuni hamba, Yang Mulia.
  "GET OUT OF THE WAY!"
      repaired: GET OUT OF THE WAY!
      Indonesian: Minggir!
  "Chapter 3: The Long Night (Part 2)"
      repaired: Chapter 3: The Long Night (Part 2)
      Indonesian: Bab 3: Malam Panjang (Bagian 2)""",
)

JAPANESE = Language(
    code="ja",
    name="Japanese",
    spaced=False,
    example_silence="黙れ。",
    ocr_damage="""\
A name that appears more than once on the page is one name: OCR corrupts \
a character in one instance and not another, so シロコ and ジロコ on the \
same page are the same person and must come back spelled the same way. \
Prefer the reading that is a real name.\
\
Expect a wrong kanji that looks like the right one, small kana (っ ゃ ゅ ょ) \
read at full size, and the long vowel mark ー confused with a dash or with \
the digit 1. Vertical lettering is read top to bottom, columns right to \
left; if a line reads as nonsense, consider that two columns may have been \
merged. Repair to the reading the sentence needs, keeping particles intact --\
dropping は, が, を or に changes who is doing what to whom.""",
)

CHINESE = Language(
    code="zh",
    name="Chinese",
    spaced=False,
    example_silence="安静。",
    ocr_damage="""\
A name that appears more than once on the page is one name: OCR corrupts \
a character in one instance and not another, so two spellings a stroke \
apart are the same person and must come back spelled the same way.\
\
Expect a wrong hanzi that looks like the right one, and simplified and \
traditional forms mixed within one line. Repair to the character the sentence \
needs. Vertical lettering is read top to bottom, columns right to left; if a \
line reads as nonsense, consider that two columns may have been merged.""",
)

LANGUAGES = {
    language.code: language
    for language in (ENGLISH, INDONESIAN, JAPANESE, CHINESE)
}


# A run of three or more of one letter is the letterer stretching a noise out
# to fit the drawing. Two is ordinary spelling -- "squeeze" has one.
_STRETCHED = re.compile(r"(.)\1{2,}")
# The same run, but of letters only. The dash a letterer ends a line with is
# written "--" and sometimes "---", and that is punctuation, not a stretch.
_STRETCHED_LETTER = re.compile(r"([a-z])\1{2,}")
# What is left of a lettered word once the marks around it are gone. The
# hyphen stays, because "BA-DUMP" is one noise and losing it would leave "ba".
_SFX_LETTERS = re.compile(r"[^a-z-]+")


def _unstretched(word: str) -> set[str]:
    """A lettered word and the ordinary spellings it could be a stretch of.

    Both collapses are tried because neither alone covers the two ways a
    letterer stretches a word: "SQUEEEEEZE" is "squeeze", where the doubled
    letter is part of the spelling, and "THUUUD" is "thud", where it is not.
    """
    return {word, _STRETCHED.sub(r"\1\1", word), _STRETCHED.sub(r"\1", word)}


def _is_sfx_word(word: str, vocabulary: frozenset[str]) -> bool:
    if _unstretched(word) & vocabulary:
        return True
    # "THUD-THUD" and "BADUMP" are the same noise written two ways, so a
    # hyphenated word counts if it is one this list knows or if every piece of
    # it is. Only one level deep, which is as far as lettering ever goes.
    if "-" not in word:
        return False
    pieces = [piece for piece in word.split("-") if piece]
    if not pieces:
        return False
    if _unstretched(word.replace("-", "")) & vocabulary:
        return True
    return all(_unstretched(piece) & vocabulary for piece in pieces)


def is_sound_effect(text: str, language: Language) -> bool:
    """Whether a whole line is nothing but noises drawn into the artwork.

    Every word has to be one, which is what keeps this from taking dialogue:
    "SQUEEZE" over a hand is a drawn effect, "squeeze it harder" is a line
    someone says, and only the first is left alone. Repetition is how a
    letterer draws a noise continuing, so "THUD THUD THUD" is still one noise.
    """
    if not language.sfx_words:
        return False
    words = [_SFX_LETTERS.sub("", word.lower()) for word in text.split()]
    words = [word.strip("-") for word in words if word.strip("-")]
    return bool(words) and all(
        _is_sfx_word(word, language.sfx_words) for word in words
    )


# A drawn noise is one word. OCR of hand lettering routinely attaches a second
# -- "GUUUURRRGLE" came back as "GUUUURRAGIE AND" -- so one stray is allowed
# for. Three separate words is a sentence, and a sentence with a stretched
# word in it is somebody shouting rather than something the artist drew.
_NOISE_WORDS = 2


def is_stretched_noise(text: str, language: Language) -> bool:
    """Whether a reading is a noise the letterer drew out to fit the picture.

    English does not spell any word with three of the same letter in a row, so
    a run of three is the letterer stretching the noise across the drawing --
    and it survives OCR damage in a way the word itself does not. That matters
    because hand lettering is exactly what OCR is worst at: "GUUUURRRGLE",
    drawn diagonally across a whole panel, was read as "GUUUURRAGIE AND".
    Neither the noise list nor the model recognised the word, so the panel was
    erased and "GUUUURRAGIE DAN" set across it in the comic face.

    On its own this is not enough -- "NOOOO! YOU DON'T UNDERSTAND!" is a line
    somebody says and stretches the same way. So the caller asks it only of
    lettering nothing on the page encloses (see `Region.is_unbounded`); a
    shout in a balloon is dialogue and stays dialogue. Over three chapters
    twelve readings carried a stretched run: the eight in balloons were all
    speech, and the four drawn onto artwork -- GUUUURRRGLE, EEEE, WOOOW,
    EEEEK! -- were all drawn.
    """
    if not language.spaced:
        # Kana and han have no letters to repeat: a Japanese effect is drawn
        # large and out of column, which is a different test entirely, and the
        # line head already makes it.
        return False
    words = [_SFX_LETTERS.sub("", word.lower()) for word in text.split()]
    words = [word for word in words if word]
    if not words or len({*words}) > _NOISE_WORDS:
        return False
    return any(_STRETCHED_LETTER.search(word) for word in words)


@dataclass(frozen=True)
class Pair:
    """A source and a target, and the guard that fits them."""

    source: Language
    target: Language

    @property
    def code(self) -> str:
        return f"{self.source.code}-{self.target.code}"

    @property
    def label(self) -> str:
        return f"{self.source.name} to {self.target.name}"


# What the settings panel offers. Ordered with the pair the tool was built for
# first, since it is still the one with the most tuning behind it.
PAIRS: list[Pair] = [
    Pair(ENGLISH, INDONESIAN),
    Pair(JAPANESE, INDONESIAN),
    Pair(JAPANESE, ENGLISH),
    Pair(CHINESE, INDONESIAN),
    Pair(CHINESE, ENGLISH),
]

PAIRS_BY_CODE = {pair.code: pair for pair in PAIRS}

DEFAULT_PAIR = PAIRS[0].code


def pair_for(code: str) -> Pair:
    """The pair named by `code`, falling back to the default.

    Falling back rather than raising: an unknown pair in a settings file is a
    typo or a downgrade, and refusing to start over it helps nobody.
    """
    return PAIRS_BY_CODE.get(code, PAIRS_BY_CODE[DEFAULT_PAIR])
