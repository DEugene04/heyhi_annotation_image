# Transcripts — ground truth for the OCR accuracy test

One transcript per image, used to measure how accurately the OCR reads the
handwriting (the recognition-ceiling spike, T2.5).

## Convention

- One file per image, **same base name**, `.txt` extension:
  `whiteboard_unskewed_clear_math_1.jpeg` → `whiteboard_unskewed_clear_math_1.txt`
- **One physical line per handwritten line**, typed as you read it, top to bottom.
- Plain text — write maths the natural way you'd read it aloud
  (`2x = 16 - 2`, `x^2 + 3x + 2`). Don't try to write LaTeX; the accuracy test
  normalises both sides before comparing.
- Only the student's own writing. Skip whiteboard glare, smudges, and anything
  you genuinely can't read (leave that line blank if a line is illegible, so the
  line count still lines up).

## Why per-line

The accuracy test compares each transcript line against the matching line the OCR
returned, and reports the share of lines read exactly. Your line breaks are the
comparison units, so put one written line per text line.

## A note on maths matching

"Exact match" is fuzzy for maths — the OCR returns LaTeX, your transcript is plain
text — so the test normalises whitespace and unwraps notation before comparing,
and still needs a human eye on the near-misses. How strict that match should be is
a tuning decision we'll make once we see the first numbers.
