// The renderer: the cleaned image with interactive annotations drawn over it,
// and a feedback panel below.
//
// Each shape follows the writing and is tinted by its category. Selecting a
// shape — or its card in the panel — emphasises every shape belonging to that
// piece of feedback and highlights its card. Feedback with no shape is listed in
// the panel on its own. The overlay is drawn in image coordinates, so it stays
// aligned with the writing as the image scales to fit the screen.

import { useEffect, useMemo, useState } from "react";
import {
  Annotorious,
  ImageAnnotator,
  useAnnotator,
  useSelection,
  type Annotator,
  type ImageAnnotation,
  type DrawingStyle,
  type Color,
} from "@annotorious/react";
import "@annotorious/react/annotorious-react.css";

import { SAMPLE_PAYLOAD } from "./fixture";
import { toAnnotations } from "./annotations";
import { CATEGORY_STYLE } from "./categories";
import { FeedbackPanel } from "./FeedbackPanel";
import type { AnnotationPayload, Category } from "./types";
import "./App.css";

function Viewer() {
  const anno = useAnnotator<Annotator<ImageAnnotation>>();
  const selection = useSelection<ImageAnnotation>();
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

  // Load the real fused payload (Mathpix shapes + GPT-5 text) if it has been
  // exported into public/; otherwise fall back to the hand-made fixture.
  const [payload, setPayload] = useState<AnnotationPayload>(SAMPLE_PAYLOAD);
  useEffect(() => {
    fetch("/fusion-payload.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setPayload)
      .catch(() => setPayload(SAMPLE_PAYLOAD));
  }, []);

  const annotations = useMemo(() => toAnnotations(payload), [payload]);

  // Load the shapes once the annotator is ready.
  useEffect(() => {
    if (anno) anno.setAnnotations(annotations);
  }, [anno, annotations]);

  // Tint every shape by its category, always. The shape(s) of the selected
  // feedback item are drawn stronger, so the whole item stands out at once.
  useEffect(() => {
    if (!anno) return;
    const style = (annotation: ImageAnnotation): DrawingStyle => {
      const category = annotation.properties?.category as Category;
      const color = (CATEGORY_STYLE[category]?.color ?? "#666666") as Color;
      const isSelected = annotation.properties?.itemId === selectedItemId;
      return {
        fill: color,
        fillOpacity: isSelected ? 0.3 : 0.15,
        stroke: color,
        strokeOpacity: isSelected ? 1 : 0.65,
        strokeWidth: isSelected ? 3 : 1.5,
      };
    };
    anno.setStyle(style);
  }, [anno, selectedItemId]);

  // A click on a shape selects its owning feedback item.
  useEffect(() => {
    const picked = selection.selected?.[0]?.annotation;
    const itemId = picked?.properties?.itemId as string | undefined;
    if (itemId) setSelectedItemId(itemId);
  }, [selection]);

  // A click on a panel card selects the item, and its shape(s) on the image.
  const selectItem = (itemId: string) => {
    setSelectedItemId(itemId);
    if (!anno) return;
    const item = payload.items.find((i) => i.id === itemId);
    if (item && item.regions.length > 0) {
      anno.setSelected(`${itemId}__0`);
    } else {
      anno.cancelSelected();
    }
  };

  return (
    <div className="viewer">
      <div className="viewer__image">
        <ImageAnnotator drawingEnabled={false}>
          <img src="/fusion-image.png" alt="Student's answer" />
        </ImageAnnotator>
      </div>
      <FeedbackPanel
        items={payload.items}
        selectedItemId={selectedItemId}
        onSelect={selectItem}
      />
    </div>
  );
}

export default function App() {
  return (
    <main className="app">
      <header className="app__header">
        <h1>Answer feedback</h1>
        <p>Select a highlight, or a card below, to see its feedback.</p>
      </header>
      <Annotorious>
        <Viewer />
      </Annotorious>
    </main>
  );
}
