# Handwritten Answer Annotation

Turns a photo of a student's handwritten answer into an interactive annotated
image: each annotated region carries feedback that appears when the student
selects it. Subjects are English, mathematics, and science.

See [development-spec.md](development-spec.md) for what we're building and the
rules for building it, and [task-breakdown.md](task-breakdown.md) for the
workstream plan.

## Pipeline

Strictly serial — each layer hands its output to the next:

```
photo → Mathpix (read) → IR (package)
      → evaluator (judge, ON HOLD) → span→geometry (shapes) → Annotorious (draw)
```

The evaluator is on hold. During development the chain closes with **fixture
feedback** in its place, so the whole pipeline is testable end-to-end without it.

## Layout

```
main.py                    /health + /annotate pipeline route
config.py                  loads settings from .env
contracts/schema.py        the IR and annotation-payload shapes (Pydantic)
contracts/evaluator_contract.md   spec to hand the org (kept separate)
ocr/                       Mathpix + IR construction (WS2)
  ocr/mathpix.py           two readings per image: line + word
  ocr/hybrid.py            combine them: line skeleton + word precision
geometry/                  span → shapes (WS3-core)
fixtures/                  stand-in evaluator feedback
frontend/                  Annotorious renderer (WS4)
tools/                     accuracy + overlay spikes (see below)
corpus/                    test photos + transcripts (see corpus/README.md)
```

## Accuracy tools (WS2 spikes)

Both read a corpus image via Mathpix — or a saved response under
`corpus/mathpix_cache/` — so they run without a key once a response is cached.

```
# How accurately is the handwriting read, line by line? (needs transcripts)
python -m tools.accuracy_spike

# Draw the OCR line-shapes back onto an image, to check they sit on the writing.
python -m tools.overlay_quads corpus/<image>
```

Transcripts (the ground truth for the accuracy test) go in `corpus/transcripts/`
— see the README there for the format.

## Setup

```
conda activate annotation_project      # Python 3.11
pip install -r requirements.txt
cp .env.example .env                   # then fill in your Mathpix credentials
```

## Run the backend

```
conda activate annotation_project
uvicorn main:app --reload
```

Then check `http://127.0.0.1:8000/health` and the interactive docs at
`http://127.0.0.1:8000/docs`.
