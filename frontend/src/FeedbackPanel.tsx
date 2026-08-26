// The feedback panel shown below the image.
//
// Every feedback item is listed as a card with a category chip and the comment.
// Items that have a shape on the page can be selected — selecting one highlights
// its shape(s) and vice versa. Items with no shape (feedback that has no place
// on the page) are listed together at the end, so nothing is ever lost just
// because it could not be drawn.

import { CATEGORY_STYLE } from "./categories";
import type { AnnotationItem } from "./types";

interface Props {
  items: AnnotationItem[];
  selectedItemId: string | null;
  onSelect: (itemId: string) => void;
}

function Card({
  item,
  selected,
  onSelect,
}: {
  item: AnnotationItem;
  selected: boolean;
  onSelect: () => void;
}) {
  const style = CATEGORY_STYLE[item.category];
  const locatable = item.regions.length > 0;

  return (
    <button
      type="button"
      className={`card${selected ? " card--selected" : ""}`}
      style={{ borderLeftColor: style.color }}
      onClick={onSelect}
    >
      <span className="chip" style={{ backgroundColor: style.color }}>
        {style.label}
      </span>
      <p className="card__comment">{item.comment}</p>
      {!locatable && <span className="card__note">No place on the page</span>}
    </button>
  );
}

export function FeedbackPanel({ items, selectedItemId, onSelect }: Props) {
  const located = items.filter((i) => i.regions.length > 0);
  const general = items.filter((i) => i.regions.length === 0);

  return (
    <div className="panel">
      <h2 className="panel__heading">Feedback</h2>

      {located.map((item) => (
        <Card
          key={item.id}
          item={item}
          selected={item.id === selectedItemId}
          onSelect={() => onSelect(item.id)}
        />
      ))}

      {general.length > 0 && (
        <>
          <h3 className="panel__subheading">General notes</h3>
          {general.map((item) => (
            <Card
              key={item.id}
              item={item}
              selected={item.id === selectedItemId}
              onSelect={() => onSelect(item.id)}
            />
          ))}
        </>
      )}
    </div>
  );
}
