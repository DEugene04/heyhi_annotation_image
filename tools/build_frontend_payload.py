"""
Builds a real annotation payload from the VLM+OCR fusion and drops it into the
frontend, so the renderer shows actual fused data instead of the hand-made fixture.

For one image it reads the page (Mathpix shapes + GPT-5 text), fuses them, attaches
a few stand-in feedback items to real spans of the text (the evaluator is on hold),
resolves those to shapes, and writes:
  frontend/public/fusion-payload.json  — the payload the renderer draws
  frontend/public/fusion-image.png     — the image it draws on

Run from the repo root:
    python -m tools.build_frontend_payload handWriting_badWriting_english
"""

import sys
from pathlib import Path

from PIL import Image

from fixtures.sample_payload import stand_in_feedback
from geometry.resolver import resolve_payload
from ocr.fusion import AlignmentError, fuse
from ocr.ir_builder import reconstruct
from tools.mathpix_cache import get_readings
from tools.vlm_cache import get_vlm_lines

CORPUS = Path("corpus")
PUBLIC = Path("frontend/public")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "handWriting_badWriting_english"
    image_path = next(p for p in CORPUS.glob(f"{Path(name).stem}.*") if p.suffix != ".txt")

    readings = get_readings(image_path)
    image = Image.open(image_path).convert("RGB")
    ir = reconstruct(readings["line"], readings["word"], image.width, image.height)
    try:
        fused = fuse(ir, get_vlm_lines(image_path))
    except AlignmentError as exc:
        # A page the readers disagree on is refused rather than annotated wrongly;
        # in the product this is where the "retake the photo" warning is shown.
        print(f"refused: {exc}")
        return
    payload = resolve_payload(fused, stand_in_feedback(fused))

    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "fusion-payload.json").write_text(payload.model_dump_json(indent=2))
    image.save(PUBLIC / "fusion-image.png")
    print(f"wrote frontend/public/fusion-payload.json "
          f"({len(payload.items)} items) and fusion-image.png "
          f"({image.width}x{image.height})")


if __name__ == "__main__":
    main()
