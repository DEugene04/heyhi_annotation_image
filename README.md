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
photo → Mathpix (shapes) ┐
        GPT-5 (text)      ├→ fuse → IR (package)
                          ┘
      → evaluator (judge, ON HOLD) → span→geometry (shapes) → Annotorious (draw)
```

The evaluator is on hold. During development the chain closes with **fixture
feedback** in its place, so the whole pipeline is testable end-to-end without it.

## How we read the page

Two readers look at each photo, and each is trusted for the one thing it does
best:

- **Mathpix** — trusted for **where** the writing is (the position and shape of
  every line and word on the page). It reads the page two ways:
  - *line reading* — how many lines there are, in reading order.
  - *word reading* — the exact shape of every individual word.

  We combine them: the line reading gives the skeleton, and the word shapes are
  dropped onto it for precise boxes. (A block of maths that Mathpix squashes into
  one box is split back into its separate rows here.)

- **GPT-5** — trusted for **what** the writing says (the actual text, including
  messy handwriting Mathpix gets wrong).

**Fusion** then lays GPT-5's text onto Mathpix's boxes: accurate words sitting in
accurate places. So every highlight the student sees carries the *right text* in
the *right spot*.

```
Mathpix line reading ─┐
                      ├─► accurate boxes ─┐
Mathpix word reading ─┘                   ├─► fusion ─► text in the right place
GPT-5 text ───────────────────────────────┘
```

## Layout

```
main.py                    /health + /annotate pipeline route
config.py                  loads settings from .env
contracts/schema.py        the IR and annotation-payload shapes (Pydantic)
contracts/evaluator_contract.md   spec to hand the org (kept separate)
ocr/                       reading the page (WS2)
  ocr/mathpix.py           call Mathpix: two readings per image (line + word)
  ocr/vlm.py               call GPT-5: the text of each line
  ocr/ir_builder.py        build the IR: line skeleton + word precision
  ocr/fusion.py            lay GPT-5's text onto Mathpix's shapes
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
