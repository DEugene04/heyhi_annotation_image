"""
Writes out, in plain text, what the hybrid OCR produced for each corpus image —
for eyeballing against the hand-typed transcripts.

For every image with a saved reading, this writes corpus/mathpix_reads/<image>.txt:
one line per reconstructed line or maths row, tagged with its type and the number
of words it holds, so grouped-maths blocks that were split back into rows are easy
to see.

Run from the repo root:
    python -m tools.dump_reads
"""

import glob
from pathlib import Path

from PIL import Image

from ocr.ir_builder import reconstruct
from tools.mathpix_cache import get_readings

CORPUS = Path("corpus")
READS_DIR = CORPUS / "mathpix_reads"


def main() -> None:
    READS_DIR.mkdir(parents=True, exist_ok=True)
    images = [p for p in CORPUS.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    for image_path in sorted(images):
        try:
            readings = get_readings(image_path)
            width, height = Image.open(image_path).size
            ir = reconstruct(readings["line"], readings["word"], width, height)
        except Exception as exc:  # e.g. Mathpix finds no text on a too-rotated page
            print(f"skip  {image_path.name}: {exc}")
            continue

        lines = [
            f"[{seg.type.value:7}] ({len(seg.words)}w) {seg.text}"
            for seg in ir.segments
        ]
        for figure in ir.diagrams:
            lines.append(f"[diagram] (figure)")

        out = READS_DIR / f"{image_path.stem}.txt"
        out.write_text("\n".join(lines) + "\n")
        print(f"wrote {out}  ({len(ir.segments)} lines/rows, {len(ir.diagrams)} figures)")


if __name__ == "__main__":
    main()
