# Test Corpus

Real handwritten-answer photos used as ground truth for every spike. The images
themselves are not committed (see .gitignore) — this folder just documents what
to collect. Aim for **30–50 photos**.

Take real phone photos, not clean scans — the whole image-cleaning layer exists
because inputs are messy. Name files by their dimensions, e.g.
`maths_skew_pencil_lined_01.jpg`, so a spike can pull "all the pencil ones" easily.

## Coverage

**Subject / content**
- Maths — the bulk (~15–20): multi-line working, fractions, exponents, roots,
  simultaneous equations. Include a few where the same line repeats (e.g. `x = 5`
  twice).
- Science — ~8–10, including 2–3 with a diagram (labelled cell, circuit, forces).
- English — ~8–10: longer prose, paragraphs, some with crossings-out.

**Capture quality** (spread across the subjects above)
- Clean, flat, well-lit (baseline).
- Skewed / photographed at an angle.
- Curled / bent paper.
- Shadowed / uneven lighting.
- A few genuinely messy (skew + shadow + curl together).

**Paper type**
- Lined / ruled paper (several).
- Blank / plain paper.
- Grid / squared paper, if used for maths.

**Writing medium**
- Pencil / faint ink (several, light strokes).
- Normal pen (baseline).

**Edge cases** (a small, high-value handful)
- Cross-outs / rewritten working.
- The question copied onto the answer sheet.
- One or two partially illegible.

## For the maths ones

Where you can, note the ground-truth text (a quick typed transcript). The
recognition-ceiling spike measures per-line exact-match rate, which needs
something to compare against.

## Minimum to start

~10 maths photos (mix of clean and skewed, a couple in pencil) with transcripts.
That alone unblocks the first spike; backfill the rest afterwards.
