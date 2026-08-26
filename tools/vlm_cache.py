"""
Gets the VLM transcription for an image, from a saved file when there is one and
the live VLM otherwise.

Saved transcriptions live in corpus/vlm_reads/<image>.txt, one line per written
line. This lets the fusion demo run without an API key (using a transcription
typed in by hand as a stand-in for GPT-5), and avoids re-reading the same image.
"""

from pathlib import Path

from ocr.vlm import transcribe_lines

VLM_DIR = Path("corpus/vlm_reads")


def get_vlm_lines(image_path: Path) -> list[str]:
    """Return the VLM's line-by-line reading of an image."""

    saved = VLM_DIR / f"{image_path.stem}.txt"
    if saved.exists():
        return [line for line in saved.read_text().splitlines() if line.strip()]

    lines = transcribe_lines(image_path.read_bytes())
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    saved.write_text("\n".join(lines) + "\n")
    return lines
