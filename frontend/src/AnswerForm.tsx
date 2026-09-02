// The submission form: a photo of the answer plus the question (required), and
// an optional answer key, rubric, and marking note. On submit it POSTs the lot
// as multipart/form-data to the backend's /annotate route and hands the returned
// payload back to the parent to draw.
//
// The backend refuses a page it cannot read reliably with HTTP 422 and a plain
// string reason (retake the photo); FastAPI's own validation errors also use 422
// but carry a structured detail, so the two are told apart by the detail's type.

import { useState, type FormEvent } from "react";
import type { AnnotationPayload } from "./types";

interface Props {
  onResult: (payload: AnnotationPayload, photo: File) => void;
}

export function AnswerForm({ onResult }: Props) {
  const [photo, setPhoto] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [questionType, setQuestionType] = useState<"short_answer" | "essay">(
    "short_answer",
  );
  const [answerKey, setAnswerKey] = useState("");
  const [rubric, setRubric] = useState("");
  const [markingNote, setMarkingNote] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!photo || !question.trim() || busy) return;

    setBusy(true);
    setError(null);

    const body = new FormData();
    body.append("photo", photo);
    body.append("question", question);
    body.append("question_type", questionType);
    if (answerKey.trim()) body.append("answer_key", answerKey);
    if (rubric.trim()) body.append("rubric", rubric);
    if (markingNote.trim()) body.append("marking_note", markingNote);

    try {
      const response = await fetch("/annotate", { method: "POST", body });
      if (response.ok) {
        onResult((await response.json()) as AnnotationPayload, photo);
      } else if (response.status === 422) {
        const detail = (await response.json())?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : "That upload was not accepted. Check the photo and question.",
        );
      } else {
        setError("Marking failed. Please try again.");
      }
    } catch {
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="form" onSubmit={submit}>
      <label className="form__field">
        <span className="form__label">Answer photo</span>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setPhoto(e.target.files?.[0] ?? null)}
          required
        />
      </label>

      <label className="form__field">
        <span className="form__label">Question</span>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={2}
          placeholder="The question the student was answering"
          required
        />
      </label>

      <label className="form__field">
        <span className="form__label">Question type</span>
        <select
          value={questionType}
          onChange={(e) =>
            setQuestionType(e.target.value as "short_answer" | "essay")
          }
        >
          <option value="short_answer">Short answer</option>
          <option value="essay">Essay</option>
        </select>
      </label>

      {questionType === "essay" ? (
        <>
          <label className="form__field">
            <span className="form__label">
              Rubric <span className="form__optional">(optional)</span>
            </span>
            <textarea
              value={rubric}
              onChange={(e) => setRubric(e.target.value)}
              rows={3}
              placeholder='A rubric table as JSON (list of criteria). Leave empty to use the default rubric.'
            />
          </label>
          <p className="form__optional">
            Leave the rubric empty to mark against the default rubric. Strengths are
            highlighted green, spelling/grammar red, and per-criterion scores appear
            in the panel.
          </p>
        </>
      ) : (
        <>
          <label className="form__field">
            <span className="form__label">
              Answer key <span className="form__optional">(optional)</span>
            </span>
            <textarea
              value={answerKey}
              onChange={(e) => setAnswerKey(e.target.value)}
              rows={2}
              placeholder="The ground-truth answer, if you have one"
            />
          </label>

          <label className="form__field">
            <span className="form__label">
              Rubric <span className="form__optional">(optional)</span>
            </span>
            <textarea
              value={rubric}
              onChange={(e) => setRubric(e.target.value)}
              rows={2}
              placeholder="Scoring criteria to follow"
            />
          </label>

          <label className="form__field">
            <span className="form__label">
              Marking note <span className="form__optional">(optional)</span>
            </span>
            <textarea
              value={markingNote}
              onChange={(e) => setMarkingNote(e.target.value)}
              rows={2}
              placeholder="Question-specific marking instructions"
            />
          </label>
        </>
      )}

      <div className="form__actions">
        <button type="submit" className="form__submit" disabled={busy || !photo || !question.trim()}>
          {busy ? "Marking…" : "Mark answer"}
        </button>
        {error && <p className="form__error">{error}</p>}
      </div>
    </form>
  );
}
