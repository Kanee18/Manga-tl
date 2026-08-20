"""Drawing text the lettering face cannot draw on its own.

Anime Ace is a comic face with a comic face's character set: capitals, digits,
the punctuation English lettering uses. It has no glyph for ☆, ★, ♪, ♡, →, or
for the Japanese comma and full stop -- and Skia does not complain about a
missing glyph, it draws `.notdef`, which in this face is nothing at all.

That is invisible until the source is Japanese, where a decorative mark inside
a line is ordinary. The translator kept the ☆ in the Indonesian exactly as it
should, the typesetter measured a width for it, laid out around it, and drew a
hole. Reported as the symbol going missing from the translation, which is
precisely what it was.

So a line is split into runs of characters one typeface can draw, and each run
is measured and drawn with that typeface. Runs are found per character against
the primary face, and anything it lacks is handed to whatever the system font
manager offers for that character -- Segoe UI Symbol for ☆ and ♪, Yu Gothic
for 、 and 。 on a Windows machine, and something equivalent elsewhere.

The alternative was to strip what could not be drawn. That is worse and not
much simpler: a line ending "さすが☆" means something slightly different from
"さすが", and silently deleting from a translation is the kind of change
nobody can see to check.
"""

from __future__ import annotations

import skia

# Sized once and reused; building a Font is cheap but measuring a page is not
# the place to find that out.
_FALLBACK_CACHE: dict[int, skia.Typeface | None] = {}


def _fallback_for(code_point: int) -> skia.Typeface | None:
    """A system typeface that can draw this character, if one exists."""
    if code_point not in _FALLBACK_CACHE:
        try:
            _FALLBACK_CACHE[code_point] = skia.FontMgr().matchFamilyStyleCharacter(
                "", skia.FontStyle(), [], code_point
            )
        except Exception:  # noqa: BLE001 -- a machine with no font manager
            _FALLBACK_CACHE[code_point] = None
    return _FALLBACK_CACHE[code_point]


class Face:
    """One lettering face, plus whatever it takes to draw what it cannot.

    Stands in for a `skia.Font` everywhere text is measured or drawn, so the
    fallback is not something each caller has to remember.
    """

    def __init__(self, typeface: skia.Typeface, size: float) -> None:
        self._size = size
        self._primary = _prepare(typeface, size)
        self._extra: dict[str, skia.Font] = {}

    @property
    def size(self) -> float:
        return self._size

    def getTypeface(self) -> skia.Typeface:  # noqa: N802 -- matches skia.Font
        """The face the lettering is set in, ignoring any fallback."""
        return self._primary.getTypeface()

    def getMetrics(self) -> skia.FontMetrics:  # noqa: N802 -- matches skia.Font
        """The primary face's metrics.

        Deliberately not the fallback's. Line height and baseline come from
        the face the lettering is set in; letting an incidental ☆ from some
        other family shift the leading would make one balloon sit differently
        from its neighbours for a reason nobody could see.
        """
        return self._primary.getMetrics()

    def _for(self, character: str) -> skia.Font:
        if self._primary.unicharToGlyph(ord(character)):
            return self._primary
        if character not in self._extra:
            found = _fallback_for(ord(character))
            self._extra[character] = (
                _prepare(found, self._size) if found is not None else self._primary
            )
        return self._extra[character]

    def runs(self, text: str) -> list[tuple[str, skia.Font]]:
        """`text` split into the longest pieces one typeface can draw."""
        pieces: list[tuple[str, skia.Font]] = []
        for character in text:
            font = self._for(character)
            if pieces and pieces[-1][1] is font:
                pieces[-1] = (pieces[-1][0] + character, font)
            else:
                pieces.append((character, font))
        return pieces

    def measureText(self, text: str) -> float:  # noqa: N802 -- matches skia.Font
        """How wide `text` is, counting each run in the face that draws it.

        Measuring the whole string against the primary face is what made the
        hole the right size: `.notdef` has an advance, so the layout reserved
        space for a character that was never drawn.
        """
        return sum(font.measureText(piece) for piece, font in self.runs(text))

    def draw(
        self, canvas: skia.Canvas, text: str, x: float, y: float, paint: skia.Paint
    ) -> None:
        for piece, font in self.runs(text):
            canvas.drawString(piece, x, y, font, paint)
            x += font.measureText(piece)


def _prepare(typeface: skia.Typeface, size: float) -> skia.Font:
    font = skia.Font(typeface, size)
    font.setSubpixel(True)
    font.setEdging(skia.Font.Edging.kAntiAlias)
    return font
