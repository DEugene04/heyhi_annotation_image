# Handwritten Answer Annotation — Task Breakdown

**Owner:** solo, full-time
**Chosen architecture:** Pipeline A — `DocRes → Mathpix → evaluator → span→geometry → Annotorious`
**Evaluator status:** **ON HOLD.** It exists and is owned by the organization; it already returns span-anchored feedback (`mark_highlights[]`), but its span format is messy and a possible replacement model (one that returns word *position*) is being checked with the supervisor. All evaluator integration is paused. Build the layers you control first.
**DocRes:** local laptop GPU for dev, org's dedicated GPU in production.

---

## What's on hold, and why it doesn't block you

The evaluator is paused pending a supervisor conversation about a position-returning model. **This blocks almost nothing.** Every layer before the evaluator — image cleanup, text recognition, IR, and the renderer — is fully buildable now, and the renderer only ever needed a payload *shape*, not a live evaluator.

Two things are parked until the evaluator resumes, and are called out inline below as **[ON HOLD]**:
- The evaluator stub and its contract.
- The span→geometry resolver's handling of the real evaluator output — specifically the duplicate-`target` collision problem (several identical `"x = 5"` spans in one answer, with no position to disambiguate them). How this is resolved depends on whether the new position-returning model materialises, so it waits.

The geometry resolver's *core* (offset-range → line-shapes → polygon) is still buildable now against fixture offsets — only the part that consumes the real evaluator's messy string-targets waits.

## The decision that shapes everything (once the evaluator is back)

**The anchoring contract is the spine of the evaluator half.** The existing evaluator anchors by verbatim `target` string, which collides on repeated spans and is corrupted by a LaTeX rewrite on maths. The supervisor's position-returning model would replace string-matching with a position lookup and make the collision problem vanish. **Which anchoring model you get determines how span→geometry is built** — so that layer's evaluator-facing half stays parked until the anchoring model is known. Don't build string-matching machinery you may be about to throw away.

---

## Workstreams

Seven workstreams. WS0–WS3 are the critical path. WS4 is parallelable. WS5–WS6 are integration and hardening.

### WS0 — Contracts & scaffolding
The data shapes every other layer reads and writes. Do this before any feature code.

- [ ] **T0.1** Define the **Intermediate Representation (IR)** schema — `segments[]` (id, line, type, text, latex, quad, char_start, char_end, confidence) + `flat_text` (line breaks preserved) + image metadata. This is §5 of the research doc. Write it as a typed schema (Pydantic / TypeScript types / JSON Schema), not prose.
- [ ] **T0.2** Define the **evaluator contract** — input: `flat_text`. Output: `[{char_start, char_end, comment, category}]`. Include a `category` enum (correct / error / info / carried…). Document it as a standalone spec file to hand to the org later.
- [ ] **T0.3** Define the **annotation payload** the frontend consumes — resolved polygons + feedback + category, plus panel-only items with no polygon.
- [ ] **T0.4** Repo scaffolding — backend service skeleton, frontend skeleton, a `/health` route, env/secrets handling for the Mathpix key, a sample-image fixture folder.
- [ ] **T0.5** Assemble a **test corpus** — 30–50 real photos: maths, science (incl. a diagram or two), English; clean and messy; flat and skewed; lined and blank paper. This is your ground truth for every spike. Cheap to start, invaluable throughout.

### WS1 — DocRes (rectify + enhance)
Runs unconditionally on every image. Local GPU now.

- [ ] **T1.1** Stand up DocRes locally — clone, weights, `inference.py` runs on your GPU. Confirm CUDA works on your laptop.
- [ ] **T1.2** Wrap it as a **callable service** (FastAPI endpoint or local module): image in → cleaned image out. Design the interface so swapping local→org-GPU later is a config change, not a rewrite.
- [ ] **T1.3** **SPIKE (pencil survival):** run `appearance` vs `binarization` on the pencil/faint-ink samples. Confirm light strokes survive. Pick the mode. *(Research doc risk: enhancement erasing faint working.)*
- [ ] **T1.4** **SPIKE (enhance on/off accuracy):** the "one-time comparison" from your flowchart — OCR the corpus with and without DocRes-enhance, compare accuracy, decide whether enhance stays. Not a runtime branch; a decision made once here.
- [ ] **T1.5** Decide `dewarp → enhance` chaining (order confirmed in research) and expose a single "clean this image" call.

### WS2 — OCR + IR construction
Turns a cleaned image into the IR. The heart of the grounding chain.

- [ ] **T2.1** Mathpix integration — `v3/text` with `include_line_data` + `include_word_data`. Handle auth, image compression (keep under ~100KB per Mathpix's own latency guidance), errors.
- [ ] **T2.2** **Parse Mathpix response → IR.** Map `cnt` quads, `type` (text/math/diagram), `is_handwritten`, `line`, `confidence` into `segments[]`. Build `flat_text` with preserved line breaks. **Construct** the fields Mathpix doesn't give you: `id`, `char_start`, `char_end`.
- [ ] **T2.3** **Printed-vs-handwritten filter** — drop printed lines (the question) using the `is_handwritten` flag; keep handwritten only. Route `diagram`-type lines to a separate track.
- [ ] **T2.4** **Confidence gate** — area-weighted confidence check; below threshold → "couldn't read clearly, retake" instead of annotating. Threshold tuned on corpus.
- [ ] **T2.5** **SPIKE (recognition ceiling):** measure per-line exact-match rate on the maths corpus. This is research risk #1 — the thing most likely to sink the UX. Know the number before committing to line-level annotation.
- [ ] **T2.6** **SPIKE (quad accuracy on skew):** overlay Mathpix quads on skewed samples, eyeball drift. Confirms whether the DocRes-cleaned image is flat enough that quads sit right.

### WS3 — Span→geometry core  +  [ON HOLD] evaluator integration
The geometry resolver's *core* is buildable now against fixture offsets. The parts that consume the **real** evaluator output are on hold until the anchoring model is known (string-`target` vs the supervisor's position-returning model).

**Buildable now (against fixtures):**
- [ ] **T3.2** **Span→geometry resolver core** — given an offset range, find overlapping IR line-shapes → contiguous runs → one polygon per run. Deterministic. This is §7. Drive it with hand-made fixture offsets, not a live evaluator.
- [ ] **T3.3** **Multi-polygon annotations** — one feedback item owning several regions (e.g. "carried from line 2"). Build it in from the start; painful to retrofit.
- [ ] **T3.4** **Span-less / panel-only items** — feedback with no resolvable span renders in the panel without a region. The safety fallback.
- [ ] **T3.5** **Span validation** — every span must resolve to real IR line-shapes; if it doesn't, drop the region and log it (never place a wrong box).

**[ON HOLD] — resumes after the supervisor conversation:**
- [ ] **T3.1** ~~Evaluator stub~~ — **HOLD.** Don't build the stub yet. Once the anchoring model is decided, build a stub that mimics the *real* response shape (`mark_highlights[]`, the swapped `remarks`/`remarks_detailed`, the LaTeX-rewritten targets) so the real service drops in with no downstream change.
- [ ] **T3.6** **[ON HOLD] `target`-string → offset locator** — the adapter that turns the evaluator's verbatim `target` strings into the offset ranges T3.2 consumes. Must handle: LaTeX normalisation (`$$..$$` vs `\(..\)`), fuzzy matching, and **the duplicate-`target` collision** (several identical `"x = 5"` in one answer). *Only needed if the anchoring stays string-based; the position-returning model would make this task disappear entirely — which is exactly why it waits.*

### WS4 — Renderer (parallelable with WS1–WS3)
Frontend. Can start as soon as T0.3 (payload shape) is locked, using fixture data.

- [ ] **T4.1** Annotorious set-up over a static image with hardcoded polygon fixtures.
- [ ] **T4.2** **Polygon annotations** (not rects) — render `quad` as SVG polygon; point-in-polygon hit-testing.
- [ ] **T4.3** **Interaction model from the Figma + Ren learnings** — persistent tint per category; border/emphasis on selection; feedback in a **panel below**, not a hover tooltip (works on touch). Consider Ren's left-margin-bar for unselected to avoid overlap clutter.
- [ ] **T4.4** **Category chips + mark display** — mirror the prototype's card (quoted span, prose, chip).
- [ ] **T4.5** **Viewport scaling** — annotations stay aligned as the image scales across screen sizes.
- [ ] **T4.6** **Panel-only items list** — render span-less feedback that has no region.

### WS5 — End-to-end integration
Wire the real chain together.

- [ ] **T5.1** Assemble the serial pipeline: upload → DocRes → Mathpix → IR → **[evaluator: on hold]** → geometry → payload → renderer. Until the evaluator resumes, close the loop with fixture feedback so the full chain is testable end-to-end without it.
- [ ] **T5.2** **Latency instrumentation** — measure each stage. Confirm the 6–8s estimate against reality; find the real bottleneck. (Research risk #5.)
- [ ] **T5.3** **Synchronous UX** — progress indicator / streamed feedback so the wait is tolerable. Consider showing the cleaned image first, then annotations as they resolve.
- [ ] **T5.4** **Error paths** — DocRes fails, Mathpix errors/times out, low confidence, zero handwritten lines detected. Each needs a defined student-facing outcome.

### WS6 — Hardening & handoff
- [ ] **T6.1** **Diagram track (v1)** — whole-figure single annotation using the diagram `cnt`. No per-element parsing. (Research §4.)
- [ ] **T6.2** **DocRes deployment abstraction** — confirm the local→org-GPU swap is config-only. Document what the org needs to host.
- [ ] **T6.3** **[ON HOLD] Evaluator handoff pack** — resumes with the evaluator. The contract spec + a stub mirroring the real response shape + the validation suite, so the org's evaluator (string-target or position-returning) can be tested against known-good cases.
- [ ] **T6.4** **Cross-cutting spikes not yet closed** — cross-out/rewritten working (research risk #10, genuinely unknown), ruled-paper handling (#4), span-less feedback proportion (#7).

---

## Critical path vs parallel

```
BUILDABLE NOW:  WS0 ──► WS1 ──► WS2 ──► WS3-core ──► WS5(fixtures) ──► WS6-partial
                          │
PARALLEL:                 └────► WS4 (renderer, on fixtures) ──────────┘

ON HOLD (resumes after supervisor conversation):
                WS3 evaluator half (T3.1, T3.6) ──► WS5 real integration ──► T6.3 handoff
```

Two independent parallel tracks feed the fixtures-based end-to-end:
- **WS4 (renderer)** only needs the T0.3 payload shape, not a backend. Build it against fixtures alongside WS1–WS2.
- **WS3-core** (geometry resolver) needs only fixture offsets, not a live evaluator.

The evaluator half of WS3 and the real end-to-end integration wait on the anchoring-model decision. Everything else runs to completion without them — you can have a full pipeline working on fixture feedback before the evaluator is touched.

---

## Risk-priority ordering (do the scary spikes early)

The research doc's risks, ordered by "how badly does this hurt if discovered late":

1. **T2.5 recognition ceiling** — if maths OCR is too inaccurate at line level, the whole annotation premise wobbles. This is completely independent of the held evaluator, so it's your **first task**, week 1.
2. **T1.3 pencil survival** — cheap to test, and a wrong DocRes mode silently destroys input.
3. **T2.6 quad-on-skew** — validates the display-on-cleaned-image decision.
4. **T5.2 latency** — you've committed to synchronous; confirm the budget holds before building elaborate UX around it. Note the held evaluator's retry loop (blocking 3s sleep + double-LLM re-run) means its worst-case latency is well beyond the happy path — factor that in when the evaluator returns.

**Deferred with the evaluator (not now):** the anchoring model (string-`target` vs position-returning) and the duplicate-`target` collision. The old "can the evaluator return offsets?" risk is *resolved* — it returns spans, not offsets — but whether those spans carry position is the open question the supervisor conversation settles.

Everything else can surface later without wrecking the plan.
