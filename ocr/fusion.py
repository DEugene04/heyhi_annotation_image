"""
Combines the VLM's accurate text with Mathpix's accurate shapes.

Mathpix gives us one segment per line (or maths row), each with a shape on the
image. The VLM reads the same page far more accurately, but splits it into lines
its own way — often merging a wrapped paragraph into one line, or breaking one
line into several. So we do not trust either reader's *line breaks*; we trust
Mathpix for *where the lines are* and the VLM for *what the words are*, and we
lay the VLM's words back onto Mathpix's lines.

How: both readers give their words in reading order. We line the two word streams
up against each other (a fuzzy, order-preserving match), which tells us, for each
of the VLM's words, the Mathpix line it belongs on — even where Mathpix misread
the word, because neighbouring words that both read the same anchor the match. We
then regroup the VLM's words by Mathpix line, so every Mathpix shape carries its
VLM text. The VLM's line breaks never enter into it, so the two disagreeing on
line count is no longer a problem.

When the two readers already split the page into the same number of lines, they
have segmented it the same way, so we skip the matching and lay each VLM line onto
the Mathpix line in the same position. This also sidesteps maths, where each line
is one LaTeX expression that does not break into comparable words — so word
matching there would be meaningless. The redistribution is only needed, and only
used, when the readers disagree on the line count.

The words only fail to line up when Mathpix's reading is genuinely broken (heavy
skew, noise) — and then its shapes are untrustworthy too, so there is nothing
worth annotating. When the counts disagree we measure how well the two agree and,
below a floor, refuse the page and ask for a clearer photo rather than annotate
it wrongly.
"""

import re
from difflib import SequenceMatcher
from typing import List, Tuple

from contracts.schema import IR, Segment

# Shown to the student when the page cannot be annotated reliably.
RETAKE_WARNING = (
    "We couldn't read this page clearly. Please retake a straight, "
    "well-lit photo of the whole answer."
)

# When the readers disagree on the line count, they must still agree on at least
# this share of the words for the redistribution to be trusted. Below it, Mathpix's
# reading is broken enough that its shapes can't be trusted either, so the page is
# refused. (When the counts match, this is not consulted — see fuse.)
_MIN_AGREEMENT = 0.30


class AlignmentError(Exception):
    """
    Raised when the two readers agree on too little of the page to fuse.

    Carries the student-facing warning to show, plus the agreement score for
    logging so a page that keeps failing can be looked at.
    """

    def __init__(self, agreement: float):
        self.agreement = agreement
        self.warning = RETAKE_WARNING
        super().__init__(
            f"{RETAKE_WARNING} (readers agree on only "
            f"{agreement:.0%} of the words)"
        )


def _norm(word: str) -> str:
    """A word reduced to letters and digits, lowercased, for matching."""

    return re.sub(r"[^a-z0-9]", "", word.lower())


def _mathpix_stream(segments: List[Segment]) -> Tuple[List[str], List[int]]:
    """
    Mathpix's words in reading order, each tagged with the line it sits on.

    The words come from each line's text, so this needs no word-shape data — only
    which line a word belongs to, which is just the line it was read from.
    """

    tokens: List[str] = []
    owners: List[int] = []
    for index, segment in enumerate(segments):
        for word in segment.text.split():
            token = _norm(word)
            if token:
                tokens.append(token)
                owners.append(index)
    return tokens, owners


def _vlm_stream(vlm_lines: List[str]) -> Tuple[List[str], List[str]]:
    """The VLM's words in reading order: the original words, and matching tokens."""

    words: List[str] = []
    tokens: List[str] = []
    for line in vlm_lines:
        for word in line.split():
            token = _norm(word)
            if token:
                words.append(word)
                tokens.append(token)
    return words, tokens


def _redistribute(ir: IR, vlm_lines: List[str]) -> Tuple[List[List[str]], float]:
    """
    Assign each VLM word to a Mathpix line, and score how well the readers agree.

    Returns the VLM words grouped by Mathpix line (one list per segment) and the
    agreement score (0-1): the share of words the two readers read the same.
    """

    segments = ir.segments
    mathpix_tokens, owners = _mathpix_stream(segments)
    vlm_words, vlm_tokens = _vlm_stream(vlm_lines)

    grouped: List[List[str]] = [[] for _ in segments]
    if not mathpix_tokens or not vlm_tokens:
        return grouped, 0.0

    # Line the VLM's words up against Mathpix's, preserving order. Each block of
    # the alignment tells us which Mathpix words a run of VLM words corresponds
    # to; from that we read off the Mathpix line each VLM word belongs on.
    matcher = SequenceMatcher(None, vlm_tokens, mathpix_tokens, autojunk=False)
    owner_of_vlm: List[int | None] = [None] * len(vlm_words)
    agreed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                owner_of_vlm[i1 + k] = owners[j1 + k]
            agreed += i2 - i1
        elif tag == "replace":
            # Same stretch read differently by each: map the VLM words across the
            # Mathpix words they replaced, in proportion, so they land on the
            # right line even with nothing matching exactly.
            span = j2 - j1
            count = i2 - i1
            for k in range(count):
                j = j1 + min(span - 1, (k * span) // count)
                owner_of_vlm[i1 + k] = owners[j]
        # "delete": VLM words with no Mathpix counterpart — left for the fill
        # below to place on a neighbour's line. "insert": Mathpix words the VLM
        # didn't produce — nothing to place.

    _fill_gaps(owner_of_vlm)

    for word, owner in zip(vlm_words, owner_of_vlm):
        grouped[owner if owner is not None else 0].append(word)

    agreement = agreed / max(len(vlm_tokens), len(mathpix_tokens))
    return grouped, agreement


def _fill_gaps(owners: List[int | None]) -> None:
    """Give every unplaced word its nearest placed neighbour's line."""

    last = None
    for k in range(len(owners)):
        if owners[k] is None:
            owners[k] = last
        else:
            last = owners[k]
    nxt = None
    for k in range(len(owners) - 1, -1, -1):
        if owners[k] is None:
            owners[k] = nxt
        else:
            nxt = owners[k]


def alignment_agreement(ir: IR, vlm_lines: List[str]) -> float:
    """How well the two readers agree on the page's words, 0-1."""

    return _redistribute(ir, vlm_lines)[1]


def alignment_ok(ir: IR, vlm_lines: List[str]) -> bool:
    """Whether the readers agree enough to fuse the page (vs. asking for a retake)."""

    if len(ir.segments) == len(vlm_lines):
        return True  # same segmentation — placed positionally, no matching needed
    return alignment_agreement(ir, vlm_lines) >= _MIN_AGREEMENT


def _texts_for(ir: IR, vlm_lines: List[str]) -> List[str]:
    """
    One VLM text per Mathpix line.

    When the readers split the page into the same number of lines, each Mathpix
    line takes the VLM line in the same position. When they differ, the VLM's
    words are redistributed onto the Mathpix lines by matching the word streams,
    and the page is refused if the two agree on too little to place them reliably.
    """

    if len(ir.segments) == len(vlm_lines):
        return list(vlm_lines)

    grouped, agreement = _redistribute(ir, vlm_lines)
    if agreement < _MIN_AGREEMENT:
        raise AlignmentError(agreement)
    return [" ".join(words) for words in grouped]


def fuse(ir: IR, vlm_lines: List[str]) -> IR:
    """
    Return a new IR with Mathpix's shapes carrying the VLM's text.

    The readers splitting the page into a different number of lines does not
    matter: the VLM's words are laid back onto Mathpix's lines (see _texts_for).
    If the readers disagree on too little to place the words reliably, raises
    AlignmentError so the page is sent back for a clearer photo.
    """

    texts = _texts_for(ir, vlm_lines)

    segments: List[Segment] = []
    offset = 0
    for segment, text in zip(ir.segments, texts):
        char_start = offset
        char_end = char_start + len(text)
        offset = char_end + 1
        segments.append(
            segment.model_copy(
                update={"text": text, "char_start": char_start, "char_end": char_end}
            )
        )

    return ir.model_copy(
        update={"segments": segments, "flat_text": "\n".join(s.text for s in segments)}
    )
