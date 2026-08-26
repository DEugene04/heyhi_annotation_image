// A hand-made payload used while the backend/evaluator chain is being built.
// The renderer only needs the payload shape, not a live backend, so this stands
// in for a real /annotate response. The shapes are positioned to sit on top of
// the writing in public/sample-answer.svg.

import type { AnnotationPayload } from "./types";

// A four-corner box, given as top-left and bottom-right, in the corner order the
// payload uses: top-left, top-right, bottom-right, bottom-left.
function box(x1: number, y1: number, x2: number, y2: number) {
  return {
    quad: [
      { x: x1, y: y1 },
      { x: x2, y: y1 },
      { x: x2, y: y2 },
      { x: x1, y: y2 },
    ],
  };
}

export const SAMPLE_PAYLOAD: AnnotationPayload = {
  image: { width: 800, height: 1000 },
  items: [
    // A correct line — one region.
    {
      id: "item-0",
      category: "correct",
      comment: "Right — the equation is copied down correctly.",
      regions: [box(78, 116, 332, 158)],
    },
    // An error on one line.
    {
      id: "item-1",
      category: "error",
      comment: "Arithmetic slip: 10 − 4 is 6, not 4.",
      regions: [box(78, 196, 232, 238)],
    },
    // One feedback item owning two separate regions: the wrong value first
    // appears here and is repeated as the final answer further down the page.
    {
      id: "item-2",
      category: "carried",
      comment:
        "This value follows from the slip above, and it is carried down into the final answer.",
      regions: [box(78, 276, 202, 318), box(78, 436, 432, 478)],
    },
    // Feedback with no place on the page — shown in the panel on its own.
    {
      id: "item-3",
      category: "info",
      comment: "Show the division step where you divide both sides by 2.",
      regions: [],
    },
  ],
};
