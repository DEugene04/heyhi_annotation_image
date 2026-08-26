// Turns our annotation payload into the shapes Annotorious draws.
//
// Annotorious treats every shape as its own annotation, but one feedback item
// can own several shapes. So each region becomes one Annotorious annotation, and
// we record which feedback item it belongs to (and its category) on the
// annotation itself. That lets a click on any shape find its feedback, and lets
// every shape of one item be emphasised together when that item is selected.

import { ShapeType, type ImageAnnotation, type Polygon } from "@annotorious/react";
import type { AnnotationPayload } from "./types";

export function toAnnotations(payload: AnnotationPayload): ImageAnnotation[] {
  const annotations: ImageAnnotation[] = [];

  for (const item of payload.items) {
    item.regions.forEach((region, index) => {
      const points = region.quad.map((p) => [p.x, p.y]);
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      const id = `${item.id}__${index}`;

      const selector: Polygon = {
        type: ShapeType.POLYGON,
        geometry: {
          points,
          bounds: {
            minX: Math.min(...xs),
            minY: Math.min(...ys),
            maxX: Math.max(...xs),
            maxY: Math.max(...ys),
          },
        },
      };

      annotations.push({
        id,
        bodies: [],
        properties: { itemId: item.id, category: item.category },
        target: { annotation: id, selector },
      });
    });
  }

  return annotations;
}
