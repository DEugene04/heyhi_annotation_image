"""
Resolves a question-image *reference* into something the LLM call can consume.

A question statement is an HTML string that may contain <img src="..."> tags. In
production those srcs point at objects in S3; in tests they point at files on
disk. The grader must not care which: it always receives a self-contained
`data:` URI (the image bytes, base64-encoded, inlined into the request) so the
model sees identical input in every environment. See DESIGN notes in
evaluator/essay/chain.py:_preprocess_question_statement, which calls this.

The one choke point is `to_data_uri(ref)`:

  - "data:..."            -> returned unchanged (already inlined).
  - "http://" / "https://"-> returned unchanged (a public URL the model can fetch
                             itself; e.g. an S3 *presigned* URL).
  - "s3://bucket/key"     -> fetched with boto3 and inlined as a data URI. Use
                             this for private objects that the model cannot reach.
  - anything else         -> treated as a local filesystem path, read, and inlined
                             as a data URI. This is what tests use.

Fetching the bytes ourselves (rather than passing a private URL through) is what
makes a local file and a private S3 object interchangeable: swap the reference
string and nothing downstream changes.
"""

import base64
import mimetypes
from pathlib import Path

# Fallback when a file's media type can't be guessed. Any raster type OpenAI
# accepts works here; the wrong-but-plausible label does not affect decoding.
_DEFAULT_MIME = "image/png"


class ImageSourceError(Exception):
    """Raised when an image reference cannot be resolved to bytes."""


def _encode(image_bytes: bytes, mime: str) -> str:
    """Inline raw image bytes as a base64 `data:` URI."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _from_s3(ref: str) -> str:
    """Download an s3://bucket/key object and inline it as a data URI."""

    try:
        import boto3  # imported here so the module loads without boto3 installed
    except ImportError as exc:  # pragma: no cover - only hit when S3 is used
        raise ImageSourceError(
            "boto3 is required to resolve s3:// image references."
        ) from exc

    without_scheme = ref[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ImageSourceError(f"Malformed S3 reference: {ref!r}")

    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
    except Exception as exc:  # noqa: BLE001 - surface any boto3/network error uniformly
        raise ImageSourceError(f"Could not fetch {ref!r} from S3: {exc}") from exc

    mime = mimetypes.guess_type(key)[0] or _DEFAULT_MIME
    return _encode(body, mime)


def _from_local(ref: str) -> str:
    """Read a local image file and inline it as a data URI."""

    path = Path(ref).expanduser()
    if not path.is_file():
        raise ImageSourceError(f"Image file not found: {ref!r}")

    mime = mimetypes.guess_type(path.name)[0] or _DEFAULT_MIME
    return _encode(path.read_bytes(), mime)


def to_data_uri(ref: str) -> str:
    """Resolve an image reference to a value usable as an <img>/image_url src.

    See the module docstring for the reference schemes. Returns the reference
    unchanged for `data:` and public `http(s)://` URLs; otherwise returns a
    base64 `data:` URI carrying the image bytes.
    """

    ref = (ref or "").strip()
    if not ref:
        raise ImageSourceError("Empty image reference.")

    lowered = ref.lower()
    if lowered.startswith("data:"):
        return ref
    if lowered.startswith(("http://", "https://")):
        return ref
    if lowered.startswith("s3://"):
        return _from_s3(ref)
    return _from_local(ref)
