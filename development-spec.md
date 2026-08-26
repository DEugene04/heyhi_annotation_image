# Handwritten Answer Annotation — Development Spec

A specification for building the image-to-interactive-annotation pipeline. This file is the source of truth for what we are building and the rules for how to build it.

---

## 1. Role

You are the engineer building this system alongside me. I am the decision-maker; you are the implementer and technical advisor. You write the code, and you flag anything uncertain before acting on it (see the Notes & Rules — this is not optional).

The system takes a photo of a student's handwritten answer and returns that image with interactive annotations drawn over specific parts of the work. Each annotated region carries its own feedback that appears when the student selects it. Subjects are English, mathematics, and science, all in English.

---

## 2. Task

Build the pipeline that turns an uploaded photo into an interactive annotated image.

The chain is a series of layers, each with one job, passing a well-defined data shape to the next. The layers we control fully are the image cleanup, the text recognition, the geometry resolution, and the renderer — **build these first.**

The evaluator (the part that judges correctness) belongs to the organization. It already exists and already returns feedback anchored to spans of the answer, so we do not need to build or change it. But **its integration is currently on hold** while we confirm how it identifies the position of each piece of feedback — the existing version points at feedback using the *text* of the span (which is ambiguous when the same text appears twice, and is altered for maths), and a different model that returns an exact *position* may be available. Until that is settled, we build every other layer and stand in for the evaluator with fixture feedback. Nothing downstream depends on which evaluator we end up with, provided the position question is resolved before we wire it in.

The priority is that the pipeline itself is correct: an image reliably becomes an accurately annotated image, with every feedback item anchored to the right region.

---

## 3. Pipeline

The flow is strictly serial — each layer waits for the one before it and hands its output forward. Every image goes through every layer; there are no shortcut branches.

```
Student photo
      │
      ▼
[1] DocRes — clean the image
      │
      ▼
[2] Mathpix OCR — read the handwriting
      │
      ▼
[3] IR builder — package what was read into a structured form
      │
      ▼
[4] Evaluator (ON HOLD — org's model) — judge correctness, using the question for context
      │
      ▼
[5] Span → Geometry — turn feedback positions into shapes on the image
      │
      ▼
[6] Annotorious renderer — draw the interactive annotations
      │
      ▼
Interactive annotated image (the cleaned image is what the student sees)
```

**Build order note.** Layers 1, 2, 3, and 6 are built now. Layer 5's *core* (turning a known position into shapes) is built now against fixture data. Layer 4 (the evaluator) and the part of Layer 5 that interprets the evaluator's real output are **on hold** — see those layer descriptions. The pipeline runs end-to-end during development by putting fixture feedback where the evaluator will go, so no layer is blocked waiting for it.

### Layer descriptions

**[1] DocRes — clean the image.**
Every uploaded photo passes through DocRes first, no exceptions. DocRes straightens the page (fixing skew, perspective, and curl from bent paper) and improves it (removing shadows and uneven lighting). The output is a clean, flat, evenly-lit version of the photo — like a scanner result. This cleaned image is the one we work with for everything after this point, and it is the one the student ultimately sees with annotations on it. We do not keep or use the original messy photo past this layer.
*Deployment note: runs on a local laptop GPU during development, and on the organization's dedicated GPU in production. The interface must be built so this swap is a configuration change, not a code rewrite.*

**[2] Mathpix OCR — read the handwriting.**
Mathpix reads the cleaned image and returns the text it sees, including maths as structured notation (LaTeX). Crucially, it returns *where* each piece of text sits on the image, as four-corner shapes that follow the writing. It reads everything on the page — including any question the student may have copied onto their answer sheet. Deciding what to ignore is not this layer's job; it reads, and passes everything forward.

**[3] IR builder — package what was read.**
This layer takes Mathpix's raw output and packages it into our own consistent structure (the "Intermediate Representation"). It records, for each line of writing: the text, the maths notation, the shape on the image, and its position as a character range within one continuous text string. It also produces that continuous text string with line breaks preserved. Everything downstream reads from this structure, not from Mathpix directly — so if we ever change OCR provider, only this layer changes.

**[4] Evaluator — judge correctness. ⏸ ON HOLD.**
This layer belongs to the organization and is paused. When it resumes, it receives the continuous text string **and the question the student was answering**. Passing the question is what lets the evaluator understand context and, importantly, recognise and exclude any question text the student copied onto their answer photo — so feedback is only ever attached to the student's actual work, never to a restated question. It returns a list of feedback items, each carrying a comment and a category (for example: correct, error, info), and each pointing at the span of the answer it refers to.
*Why it's on hold:* the way it points at a span is the open question. The existing version points using the **text** of the span — which is ambiguous when the same text (for example, "x = 5") appears more than once in an answer, and which is additionally altered for maths by an internal formatting step, so the text will not always match what we sent in. A different model that returns the **exact position** of the span may be available instead; if it is, the ambiguity disappears. We do not build this layer, or the part of Layer 5 that depends on its exact output, until this is settled. During development, fixture feedback stands in its place.
*Consequence to carry through (applies whichever evaluator we get):* because OCR transcribed the whole page, any question text the evaluator excludes still exists in the IR. That exclusion decision must reach the renderer, so we never draw an empty region over a line the evaluator deliberately ignored.

**[5] Span → Geometry — turn feedback positions into shapes.**
This layer is deterministic (no AI, no guessing). Its **core**, built now: given a position within the answer text, find which line-shapes from the IR fall there and combine them into the region(s) to draw. One feedback item can produce several regions (for example, an error that spans two separate lines). Some feedback has no region at all (for example, "you skipped a step") — that is allowed, and it renders as a panel entry with no box. If a position does not match any real text, we drop the region rather than draw it in the wrong place. This core is built and tested now using fixture positions, independent of the evaluator.
*On hold (with Layer 4):* the front part of this layer that takes the **real** evaluator's output and turns it into a position. If the evaluator points by span-text, this part must handle the maths formatting mismatch and the case where the same text appears more than once; if the evaluator points by exact position, this part is trivial or unnecessary. Because these are opposite amounts of work, we do not build it until we know which evaluator we have.

**[6] Annotorious renderer — draw the interactive annotations.**
The frontend takes the cleaned image and the resolved regions and draws them as an overlay. Each region is a shape following the writing, tinted by category. Selecting a region shows its feedback in a panel. Regions are shapes that follow the writing (not plain rectangles), so they sit correctly on the page. Feedback with no region is listed in the panel on its own.

---

## 4. Notes & Rules

These apply to every piece of work on this project.

### Code quality
- **Fix problems at the root cause.** Never write new code to work around or hide a problem in existing code. If something is broken, fix the thing that is broken.
- **Prefer updating existing code over adding new code.** When making a change, first look at whether existing code should be modified. Only add new code when the change genuinely cannot be made by updating what is already there.

### Decisions and uncertainty
- **Never assume.** Any decision that is uncertain, or that carries risk, must be raised with me explicitly and decided by me — not chosen silently and moved past. If you are unsure, stop and ask.

### Comments and documentation
- **Explain concept and purpose, simply.** Comments should describe what a piece of code is for and what it does, in plain language a non-developer could follow.
- **No history in comments.** Do not document how a function came to be, what it used to do, or what changed. Documentation is about the current concept and purpose, not the past.
