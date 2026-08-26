# Frontend — Annotorious renderer

The renderer (WS4). Takes the cleaned image and the annotation payload and draws
the regions as an interactive overlay: each region is a shape that follows the
writing, tinted by category; selecting one shows its feedback in a panel below,
and highlights every shape belonging to that piece of feedback. Panel-only
feedback (items with no region) is listed on its own under "General notes".

Built with Vite + React + TypeScript and `@annotorious/react`. It runs entirely
against a fixture payload (`src/fixture.ts`) drawn over a placeholder answer
image (`public/sample-answer.svg`) — it needs only the payload shape, not a live
backend. When the backend is wired in, replace the fixture with a fetch to
`/annotate`; the payload shape is the same (`src/types.ts` mirrors
`contracts/schema.py`).

## Run

    npm install
    npm run dev        # then open the printed local URL

## What each file does

- `types.ts` — the payload shape, mirroring the backend contract.
- `fixture.ts` — the stand-in payload, positioned over the sample image.
- `categories.ts` — the colour and label for each feedback category.
- `annotations.ts` — turns the payload into Annotorious shapes, one per region,
  each tagged with its owning feedback item.
- `FeedbackPanel.tsx` — the cards below the image, split into located feedback
  and general (panel-only) notes.
- `App.tsx` — draws the image, tints and selects shapes, and links them to the
  panel in both directions.
