"""
The recognition-ceiling spike (T2.5): how accurately does the OCR read the
handwriting, line by line?

For every corpus image that has a transcript, this reads the image (via Mathpix,
or a saved response), pulls out the handwritten lines, and compares each against
the matching transcript line. It reports the share of lines read exactly, plus a
closeness score for the near-misses, so we know the ceiling before committing to
line-level annotation.

Run from the repo root:
    python -m tools.accuracy_spike

Maths is compared leniently — spacing and LaTeX delimiters are ignored — because
the OCR returns LaTeX and the transcript is plain text. Read the near-misses by
eye; the exact-match number is a floor, not the whole story.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from contracts.schema import SegmentType
from ocr.ir_builder import reconstruct
from tools.mathpix_cache import get_readings

CORPUS = Path("corpus")
TRANSCRIPTS = CORPUS / "transcripts"

_DELIMITERS = re.compile(r"\\\(|\\\)|\\\[|\\\]|\$\$|\$")
_ENVIRONMENT = re.compile(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?")
_FRACTION = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_SPACING = re.compile(r"\\(left|right|,|;|:|!|quad|qquad| )")


def normalise(line: str) -> str:
    """Reduce a text line to what we compare: no delimiters, no surplus spacing."""

    without_math = _DELIMITERS.sub(" ", line)
    return re.sub(r"\s+", " ", without_math).strip()


def normalise_math(line: str) -> str:
    """
    Reduce a maths line to a plain form, so writing that means the same thing
    compares equal even when the OCR gives LaTeX and the transcript gives plain
    text.

    It rewrites the differences we actually see — fractions, roots, dots, powers,
    equation environments — but deliberately leaves genuine misreadings alone
    (a minus read as \\times stays different), so real errors still show up.
    """

    s = _DELIMITERS.sub(" ", line)
    s = _ENVIRONMENT.sub(" ", s)          # drop \begin{array}.. / \end{..}
    s = s.replace("\\\\", " ").replace("&", " ")  # row breaks, alignment marks
    s = _SPACING.sub(" ", s)              # \left \right \, \; thin spaces
    for _ in range(4):                    # \frac{a}{b} -> a/b, innermost first
        s, changed = _FRACTION.subn(r"\1/\2", s)
        if not changed:
            break
    s = _SQRT.sub(r"√\1", s)              # \sqrt{a} -> √a
    s = re.sub(r"\\(ldots|cdots|dots)", "...", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\(log|ln|sin|cos|tan|lim|exp)", r"\1", s)  # \log -> log
    s = s.replace("{", "").replace("}", "")   # x^{2} -> x^2, _{6} -> _6
    s = re.sub(r"\\[a-zA-Z]+", "", s)     # drop any remaining LaTeX commands
    s = s.replace("·", "*").replace("×", "*")
    return re.sub(r"\s+", "", s)          # maths ignores spacing entirely


def normalise_line(text: str, is_math: bool) -> str:
    return normalise_math(text) if is_math else normalise(text)


def closeness(a: str, b: str) -> float:
    """A 0–1 score of how similar two lines are, for reading near-misses."""

    return SequenceMatcher(None, a, b).ratio()


def check_image(image_path: Path, transcript_path: Path) -> tuple[int, int]:
    """Compare one image's reading against its transcript; return (exact, total)."""

    expected = [ln for ln in transcript_path.read_text().splitlines() if ln.strip()]
    readings = get_readings(image_path)
    width, height = Image.open(image_path).size
    ir = reconstruct(readings["line"], readings["word"], width, height)
    read = [(seg.text, seg.type == SegmentType.MATH) for seg in ir.segments]

    print(f"\n{image_path.name}")
    if len(read) != len(expected):
        print(f"  ⚠ line count differs: read {len(read)}, transcript {len(expected)}"
              " — lines are compared in order, so counts drifting will hurt matches.")

    exact = 0
    for i, want in enumerate(expected):
        got, is_math = read[i] if i < len(read) else ("", False)
        n_want, n_got = normalise_line(want, is_math), normalise_line(got, is_math)
        if n_want == n_got:
            exact += 1
        else:
            print(f"  ✗ line {i + 1}  ({closeness(n_want, n_got):.0%} close)")
            print(f"      want: {n_want}")
            print(f"      got:  {n_got}")

    print(f"  → {exact}/{len(expected)} lines exact")
    return exact, len(expected)


def main() -> None:
    pairs = []
    for transcript in sorted(TRANSCRIPTS.glob("*.txt")):
        images = list(CORPUS.glob(f"{transcript.stem}.*"))
        images = [p for p in images if p.suffix.lower() != ".txt"]
        if images:
            pairs.append((images[0], transcript))

    if not pairs:
        print("No image/transcript pairs found. Add transcripts under "
              f"{TRANSCRIPTS}/ named to match the images.")
        return

    total_exact = total_lines = 0
    for image_path, transcript in pairs:
        try:
            exact, lines = check_image(image_path, transcript)
        except Exception as exc:  # a missing key or cache surfaces here
            print(f"\n{image_path.name}\n  skipped: {exc}")
            continue
        total_exact += exact
        total_lines += lines

    if total_lines:
        print(f"\n=== overall: {total_exact}/{total_lines} lines exact "
              f"({total_exact / total_lines:.0%}) ===")


if __name__ == "__main__":
    sys.exit(main())
