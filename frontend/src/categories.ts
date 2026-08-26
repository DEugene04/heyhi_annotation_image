// How each feedback category looks: the colour used to tint its shapes and its
// chip, and the label shown to the student.

import type { Category } from "./types";

export const CATEGORY_STYLE: Record<Category, { color: string; label: string }> = {
  correct: { color: "#15803d", label: "Correct" }, // green
  error: { color: "#b91c1c", label: "Error" }, // red
  info: { color: "#1d4ed8", label: "Note" }, // blue
  carried: { color: "#b45309", label: "Carried" }, // amber
};
