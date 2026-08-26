// The shape of the annotation payload the backend produces and the frontend
// draws. It mirrors contracts/schema.py on the backend — keep the two in step.

export type Category = "correct" | "error" | "info" | "carried";

export interface Point {
  x: number;
  y: number;
}

// A region is one shape to draw, its corners following the writing.
export interface Region {
  quad: Point[];
}

// One piece of feedback. It may own several regions, one, or none. An item with
// no regions is feedback that has no place on the page — shown in the panel only.
export interface AnnotationItem {
  id: string;
  category: Category;
  comment: string;
  regions: Region[];
}

export interface AnnotationPayload {
  image: { width: number; height: number };
  items: AnnotationItem[];
}
