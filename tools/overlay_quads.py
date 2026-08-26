"""
The quad-on-skew spike (T2.6): draw the OCR's line-shapes back onto the image so
we can see, by eye, whether they sit squarely on the writing.

If the shapes drift off the words, the cleaned image wasn't flat enough and the
annotations would land in the wrong place. Handwritten lines are outlined in
green, drawn figures in blue.

Run from the repo root:
    python -m tools.overlay_quads corpus/<image>
The overlaid copy is written next to a chosen output path (default: alongside,
suffixed '.overlay.png').
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ocr.hybrid import reconstruct
from tools.mathpix_cache import get_readings


def overlay(image_path: Path, out_path: Path) -> None:
    readings = get_readings(image_path)
    image = Image.open(image_path).convert("RGB")
    ir = reconstruct(readings["line"], readings["word"], image.width, image.height)
    draw = ImageDraw.Draw(image)

    # Words first, thin orange, so the line/row outlines sit on top of them.
    word_count = 0
    for segment in ir.segments:
        for word in segment.words:
            draw.polygon([(p.x, p.y) for p in word.quad], outline=(234, 130, 40), width=1)
            word_count += 1

    for segment in ir.segments:  # each line or maths row, green
        draw.polygon([(p.x, p.y) for p in segment.quad], outline=(21, 128, 61), width=3)

    for figure in ir.diagrams:  # drawn figures, blue
        draw.polygon([(p.x, p.y) for p in figure.quad], outline=(29, 78, 216), width=3)

    image.save(out_path)
    print(f"wrote {out_path}  ({len(ir.segments)} rows, {word_count} words, "
          f"{len(ir.diagrams)} figures)")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m tools.overlay_quads <image> [output]")
        return 1
    image_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else image_path.with_suffix(
        ".overlay.png"
    )
    overlay(image_path, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
