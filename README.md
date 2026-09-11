# Handwritten Answer Annotation

Take a photo of a student's handwritten answer, and this service marks it and
hands back an **interactive image**: coloured highlights sit on the writing, and
tapping a highlight shows the feedback for that spot. It works for English,
mathematics, and science.

Think of it as an automatic teaching assistant that reads the page, grades it,
and points at exactly what was right or wrong.

---

## How it works, step by step

A photo goes in one end; an annotated, gradeable image comes out the other. In
between there are five steps, and they always run in the same order.

**1. Read the page — twice.**
Two different readers look at the same photo, because each is good at a
different thing:

- **Mathpix** is trusted for *where* the writing is — the exact position and
  shape of every line and word.
- **A vision AI model** is trusted for *what* the writing says — the actual
  words, even when the handwriting is messy.

**2. Combine the two readings ("fusion").**
We take the accurate text from the AI model and lay it onto the accurate boxes
from Mathpix. The result: the *right words* sitting in the *right places*, so
every highlight later lands on the correct spot.

**3. Safety check — is the reading trustworthy?**
If the two readers disagree, or the handwriting is just too unclear to be sure,
the service does **not** guess. It stops and asks the student for a clearer
photo. Better no annotation than a wrong one.

**4. Mark the answer.**
The service grades the reading against the question (and, if provided, an answer
key or rubric) and writes feedback. How it marks depends on the question type —
see the two modes below.

**5. Draw it.**
The feedback is turned into shapes placed on the image, and the frontend renders
them as tappable highlights.

```
 photo ─► read it (Mathpix = where, AI = what) ─► combine ─► trustworthy?
                                                                 │
                                          no ──► ask for a clearer photo
                                                                 │ yes
                                                    mark it ─► draw highlights
```

---

## Two ways to mark

The caller says which one to use with a `question_type` setting.

### Short answer  (the default)
For fill-in-the-blanks, one-line answers, rephrasing, and sentence-rewriting.

- Marks the answer against the question and an optional answer key.
- Highlights each part of the answer as correct or incorrect, with feedback that
  explains why — including any spelling or grammar mistakes that matter.

### Essay
For longer compositions marked against a rubric.

- Gives a **scorecard**: a score and comment for each rubric criterion (ideas,
  organisation, language, and so on).
- Adds **on-page spelling and grammar corrections** from a dedicated checker
  (called **FLC**), shown as highlights right on the writing.

> **Why the difference?** For short answers, the marker already points out every
> spelling and grammar slip itself, so a separate spell-checker just repeated it
> (and sometimes flagged non-issues). For essays, the marker only gives an
> overall scorecard, so the dedicated checker adds real value by marking each
> mistake in place. We measured both before deciding.

---

## What's in the project

```
main.py                the service: the /annotate route that runs the pipeline
config.py              loads settings and keys from a .env file
contracts/schema.py    the shapes of the data passed between steps

ocr/                   reading the page
  mathpix.py             Mathpix: where every line and word sits
  vlm.py                 the vision AI: what the writing says
  ir_builder.py          builds the line-by-line skeleton of the page
  fusion.py              lays the AI's text onto Mathpix's boxes

evaluator/             marking the answer
  short_answer/          the short-answer marker
  essay/                 the essay (rubric) marker
  flc.py                 the FLC spelling/grammar checker (essays only)
  flc_adapter.py         turns FLC's findings into on-page highlights

geometry/              turns feedback into shapes placed on the image
frontend/              draws the tappable highlights
tools/utils.py         shared helper used by the markers
corpus/                sample photos for testing (see corpus/README.md)
logs/                  a timestamped log saved for every run
```

For more background, see [development-spec.md](development-spec.md) (what we're
building and why) and [task-breakdown.md](task-breakdown.md) (the plan).

---

## For developers: setup and run

```
conda activate annotation_project      # Python 3.11
pip install -r requirements.txt
cp .env.example .env                    # then fill in your Mathpix + OpenAI keys
```

Start the backend:

```
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/health` to confirm it's up, and
`http://127.0.0.1:8000/docs` for the interactive API docs (where you can try
`/annotate` by uploading a photo).
