# Evaluator Contract

This document defines what the evaluator layer receives and what it must return.
It is a specification to hand to the organization that owns the evaluator — not
code we run on our side.

**Status: on hold.** The evaluator exists and is owned by the organization. Its
integration is paused while we confirm how it points at the position of each
piece of feedback (see "The open question" below). During development we stand
in for it with fixture feedback, so nothing else waits on it.

---

## What the evaluator receives

- **`flat_text`** — the student's answer as one continuous string, with line
  breaks preserved. This comes straight from our Intermediate Representation.
- **`question`** — the question the student was answering.

The question is passed for two reasons: so the evaluator understands the context
of the answer, and so it can recognise and exclude any question text the student
copied onto their answer sheet. Feedback must only ever attach to the student's
own work, never to a restated question.

## What the evaluator returns

A list of feedback items. Each item has:

| Field        | Meaning                                                        |
|--------------|----------------------------------------------------------------|
| `char_start` | Where the referenced span begins in `flat_text`.               |
| `char_end`   | Where the referenced span ends in `flat_text`.                 |
| `comment`    | The feedback to show the student.                              |
| `category`   | The kind of remark (see below).                                |

### Category values

- `correct` — this part is right.
- `error` — this part is wrong.
- `info` — a neutral note (for example, "you skipped a step").
- `carried` — a mistake carried forward from an earlier line.

An item may legitimately have no resolvable span (for example, "you skipped a
step"). Such an item still carries a comment and category; it is shown in the
panel on its own, with nothing drawn on the image.

---

## The open question (why this is on hold)

This contract asks the evaluator to point at a span by **character position**
(`char_start` / `char_end`). That is the shape we want.

The evaluator that exists today does not do this yet. It points at a span by its
**verbatim text**, which is a problem in two ways:

1. The same text (for example, `x = 5`) can appear more than once in one answer,
   so the text alone cannot say which occurrence is meant.
2. For maths, an internal formatting step rewrites the text, so what comes back
   will not always match what we sent in.

A different model that returns an exact position may be available instead. If it
is, both problems disappear and this contract is met directly. Until this is
settled, we do not build the adapter that would translate verbatim-text spans
into positions, because the position-returning model would make that work
unnecessary.
