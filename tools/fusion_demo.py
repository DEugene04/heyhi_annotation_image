"""
Demonstrates VLM + OCR fusion: Mathpix's shapes carrying the VLM's text.

For each image it reads the page with Mathpix (shapes + Mathpix's text) and with
the VLM (text only), fuses them, and shows the result side by side so the text
improvement is visible while the geometry stays put. It writes, under
corpus/fusion/:
  <image>.txt   — the fused reading (VLM text, one line per Mathpix shape)
  <image>.png   — the image with Mathpix's line/row shapes drawn on it

Run from the repo root, naming images (base names):
    python -m tools.fusion_demo handWriting_badWriting_english
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ocr.fusion import alignment_ok, fuse
from ocr.hybrid import reconstruct
from tools.mathpix_cache import get_readings
from tools.vlm_cache import get_vlm_lines

CORPUS = Path("corpus")
OUT = CORPUS / "fusion"


def run(image_path: Path) -> None:
    readings = get_readings(image_path)
    image = Image.open(image_path).convert("RGB")
    ir = reconstruct(readings["line"], readings["word"], image.width, image.height)
    vlm_lines = get_vlm_lines(image_path)
    fused = fuse(ir, vlm_lines)

    print(f"\n=== {image_path.name} ===")
    if not alignment_ok(ir, vlm_lines):
        print(f"  ⚠ line counts differ: Mathpix {len(ir.segments)} vs VLM "
              f"{len(vlm_lines)} — matched in order; refine alignment if it drifts.")
    for mathpix_seg, fused_seg in zip(ir.segments, fused.segments):
        print(f"  Mathpix: {mathpix_seg.text}")
        print(f"  GPT-5  : {fused_seg.text}")
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    OUT.joinpath(f"{image_path.stem}.txt").write_text(
        "\n".join(f"[{i + 1}] {s.text}" for i, s in enumerate(fused.segments)) + "\n"
    )
    # Draw each Mathpix box and number it, so each box can be matched to its
    # numbered GPT-5 line above and checked against the writing underneath.
    draw = ImageDraw.Draw(image)
    for i, seg in enumerate(fused.segments):
        points = [(p.x, p.y) for p in seg.quad]
        draw.polygon(points, outline=(21, 128, 61), width=3)
        top_left = min(points, key=lambda p: (p[1], p[0]))
        draw.text((top_left[0] + 3, top_left[1] + 2), str(i + 1), fill=(200, 30, 30))
    image.save(OUT / f"{image_path.stem}.png")
    print(f"  wrote corpus/fusion/{image_path.stem}.txt and .png")


def main() -> None:
    names = sys.argv[1:]
    if not names:
        print("usage: python -m tools.fusion_demo <image-base-name> [...]")
        return
    for name in names:
        image = next(p for p in CORPUS.glob(f"{Path(name).stem}.*") if p.suffix != ".txt")
        run(image)


if __name__ == "__main__":
    main()
