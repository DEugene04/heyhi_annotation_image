# Evaluator Redesign: Analyze-then-Mark (v2)

**Status: proposed.** This document is the design plan for reworking the
evaluator from its inherited "mark-then-present" shape into an "analyze-then-mark"
shape, so that the on-page highlights are strict about spelling, grammar,
argument/structure, and content — not biased toward "correct".

It is an internal engineering plan, not the org-facing spec (see
`contracts/evaluator_contract.md` for that).

---

## 1. Why (the problem)

The current evaluator is lenient by design — it was built for a content-marking
platform, not for proofreading free writing. Three mechanisms compound:

1. **Mark-first, status-echoes-mark.** The breakdown stage sets each sentence's
   status *from* the numeric mark (`get_automarking_breakdown`,
   [chain.py:520-531](chain.py#L520-L531)). A holistic "looks fine overall" mark
   drags every highlight green with it.
2. **Surface-error penalties are rubric-gated.** The marking prompt only
   penalizes spelling/grammar/structure when a rubric demands it — and defaults
   the other way (e.g. Science: "any typo, spelling, punctuation, or grammatical
   errors should not be penalized as long as the meaning stays the same",
   [logic.py:641](logic.py#L641); English grammar rules live under "Special rules
   (depending on the rubric)", [logic.py:606](logic.py#L606)).
3. **The pipeline sends no rubric/marking note by default**, so the model runs in
   its most lenient mode.

The root cause is a **dependency direction**: status is downstream of the mark. A
single holistic mark is structurally lenient; an itemized error list is
structurally strict. We fix it by reversing the dependency.

### Design principles

- **Highlights are the product.** This is an annotation tool; the per-sentence
  findings drawn on the page are primary. The numeric mark is secondary.
- **Analyze first, mark second.** The mark must follow the findings, never
  generate them.
- **Strict by default, not by opt-in.** Surface and structural errors are flagged
  unless explicitly told otherwise, the inverse of today's default.

---

## 2. Architecture

Two LLM calls, reversed and re-scoped. Same call count as today (cost neutral),
so `gpt-4.1-mini` stays on both stages (see §7).

```
question ─┐
flat_text ─┼─▶ [Stage 1: STRICT ANALYZER] ─▶ findings ─┬─▶ mark_highlights ─▶ adapter ─▶ resolver
answer_key ┘        (independent, strict)               │
rubric/note ┘                                           └─▶ [Stage 2: MARKER] ─▶ mark + reason
                                                              (thin, respects findings)
```

### Stage 1 — Strict Analyzer (runs first)

- **Input:** question, student `flat_text`, optional answer key, rubric, marking
  note.
- **Job:** independently scan the answer sentence-by-sentence and report every
  spelling, grammar, punctuation, argument-placement/structure, and content
  defect. No mark exists yet, so there is nothing to anchor leniency to. Content
  correctness uses the answer key when present; when absent, judge on the
  question and general correctness.
- **Output (this is what the frontend draws):** a list of findings, one per
  sentence/part:
  ```json
  {
    "target": "<exact sentence from the student answer>",
    "status": "correct | partially_correct | incorrect",
    "issues": ["spelling" | "grammar" | "punctuation" | "structure" | "content"],
    "remark": "<student-facing explanation>"
  }
  ```
  plus an overall `remarks` / `remarks_simplified` summary of the writing.

### Stage 2 — Marker (runs second)

- **Input:** the findings from Stage 1, `full_mark`, increment/step.
- **Job:** assign a numeric mark and a one-line justification that is **consistent
  with the findings** — it may not re-litigate or soften them. Its prompt is thin
  and carries none of the lenient subject rules from `logic.py`.
- **Output:** `{ "mark": <number>, "reason": "<justification>" }`.

### Why not just swap the existing two calls

The existing breakdown stage is a *presenter* — it takes an existing `mark` +
`reason` and rephrases them ([MarkingResultContext](logic.py#L1115)); it has
nothing to do first. And the existing marking stage carries the leniency, so
leaving it downstream lets it re-soften findings at the final step. v2 therefore
**rewrites both prompts** rather than reordering the current ones.

---

## 3. Output contract (kept stable)

`evaluate()` returns the **same dict shape** as today, so the adapter, `main.py`,
and the frontend are unaffected:

```
mark, reason, remarks, remarks_detailed, mark_highlights[], total_tokens, total_cost, models_used
```

- `mark_highlights` now come straight from Stage 1 (with the new optional
  `issues` field, which the adapter may ignore for now).
- The status vocabulary is unchanged (`correct` / `partially_correct` /
  `incorrect`), so `evaluator/adapter.py` needs **no change**: `partially_correct
  → INFO` stays (per product decision), and strictness surfaces as genuinely more
  `incorrect` (→ red ERROR) findings, not as a remapping.
- The whitespace-tolerant span locator (`_locate`) is unchanged.

---

## 4. Code shape

Build v2 as a **new, lean module** rather than mutating the large vendored
`logic.py` / `chain.py`. This isolates risk and keeps the smartjen HTTP endpoints
(`api_automarking.py`) working untouched.

- **New:** `evaluator/strict.py`
  - `async analyze(question, flat_text, answer_key, rubric, marking_note, *, api_key, model) -> AnalysisResult`
  - `async mark(findings, full_mark, step, *, api_key, model) -> MarkResult`
  - Reuses `DynamicAsyncOpenAI` (the shim in `dynamic_llm_call.py`) for the call +
    usage/cost accounting.
  - Two module-level prompt constants: `ANALYZER_SYSTEM_PROMPT`,
    `MARKER_SYSTEM_PROMPT` (drafted in §5, reviewed before wiring).
- **Changed:** `evaluator/evaluate.py`
  - `evaluate()` orchestrates `analyze()` → `mark()` and assembles the same output
    dict. The `_build_question_context` / `AutoMarkingChain` path is retired from
    this function (the classes remain for the smartjen endpoints).
- **Unchanged:** `evaluator/adapter.py`, `main.py`, all of `frontend/`.

---

## 5. Draft prompts (for review before wiring)

> These are drafts to calibrate strictness wording; exact text is Phase 0.

**Analyzer (Stage 1) — key directives:**
- "You are a strict proofreader and marker of a student's answer. Report every
  defect; do not overlook surface errors because the meaning is clear."
- "Segment the answer into sentences. For each, decide `status` and list `issues`
  from: spelling, grammar, punctuation, structure (argument placement / ordering
  / cohesion), content (factually wrong or off-question)."
- "A sentence with any spelling or grammar error is at best `partially_correct`;
  a sentence that is factually wrong, off-question, or breaks the required
  structure is `incorrect`. Reserve `correct` for sentences with no defects."
- "Use the answer key (if given) only to judge content correctness; judge
  spelling/grammar/structure independently of it."
- "Quote each `target` verbatim from the student's answer — do not fix or
  normalize it." (Required for the span locator.)
- "Be conservative about single-character oddities that may be transcription
  noise rather than the student's error" (OCR-drift guard, see §8).
- Output the findings JSON + an overall `remarks` / `remarks_simplified`.

**Marker (Stage 2) — key directives:**
- "You are given a list of findings about a student's answer and a maximum score.
  Assign a mark consistent with the findings."
- "You may not overturn a finding. If findings report errors, the mark must
  reflect them; do not restore credit on the grounds that meaning is clear."
- "Apply the increment/step and bounds [0, full_mark]."
- Output `{ mark, reason }`.

---

## 6. Adapter / frontend impact

None required. The `issues` field is additive and initially ignored by the
adapter. A future enhancement (out of scope here) could tint or group highlights
by issue type, but the current `status → Category` mapping is retained.

---

## 7. Model

`gpt-4.1-mini` on both stages (per decision). **Caveat:** mini is a mediocre
independent grammar/structure critic, so the accuracy ceiling of the strict
analyzer is set by the model — expect some false positives and misses. The stage
split makes it cheap to raise only the analyzer's model later if it specifically
underperforms, without touching the marker.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| **OCR drift misread as a student error.** `flat_text` is a reading, not the source; the analyzer could flag a transcription error as the student's spelling mistake. | Prompt guard to be conservative on single-character oddities; consider passing per-segment confidence later and suppressing low-confidence spans. Flag prominently in validation. |
| **Model strictness ceiling** (mini). | Accept for now; per-stage model bump is a one-line change if needed. |
| **False positives** (over-flagging) annoy users. | Tune analyzer wording in Phase 3 against real corpus samples; measure precision, not just recall. |
| **Mark ↔ highlight inconsistency** (marker disagrees with findings). | Marker prompt forbids overturning findings; highlights come from Stage 1 only, never regenerated from the mark. |
| **Regression for smartjen endpoints.** | v2 is a separate module; `AutoMarkingChain` and `api_automarking.py` are untouched. |

---

## 9. Implementation phases

- **Phase 0 — Prompts.** Finalize `ANALYZER_SYSTEM_PROMPT` and
  `MARKER_SYSTEM_PROMPT`; review strictness wording. *(No wiring.)*
- **Phase 1 — Strict module.** Build `evaluator/strict.py` (`analyze`, `mark`)
  with unit tests using a stubbed LLM (assert orchestration, JSON parsing, usage
  accounting). No network.
- **Phase 2 — Wire `evaluate()`.** Switch `evaluate()` to `analyze()` → `mark()`,
  keeping the output dict identical. Update `evaluate()`'s own tests. Optionally
  gate behind a `strict=True` default with the old path reachable for comparison.
- **Phase 3 — Live validation.** Run against `corpus/` samples with known
  errors; measure catch rate and false-positive rate; tune the analyzer prompt.
- **Phase 4 — Cleanup.** Remove the v1 marking path from `evaluate()` once v2 is
  trusted (leaving `AutoMarkingChain` for the smartjen endpoints).

### Test plan
- Unit (offline): stubbed-LLM orchestration in `strict.py`; adapter unchanged
  tests still pass; `main.py` route tests still pass (stub `evaluate`).
- Evaluation (live, manual): a small labeled set of answers with deliberate
  spelling/grammar/structure/content errors → confirm each is flagged with the
  right status.

---

## 10. Open questions

- Should the overall `remarks` be produced by the analyzer (consistent with
  highlights) or the marker (consistent with the number)? *Proposed: analyzer.*
- Do we ever want the numeric mark at all in the annotation UI, or only the
  highlights? If only highlights, Stage 2 could later become optional.
- Should `issues` eventually drive distinct on-page tints (e.g. spelling vs.
  content)? Out of scope now; the field is captured so we can decide later.
