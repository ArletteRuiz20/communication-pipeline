"""
ocr_utils.py
------------
Reusable OCR utility powered by Gemini Vision.
Supports PNG, JPEG, and scanned PDF files.
No extra API keys or libraries required beyond what your project already uses.
"""

import os
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _encode_file_as_base64(file_bytes: bytes) -> str:
    """Encode raw bytes to a base64 string."""
    return base64.standard_b64encode(file_bytes).decode("utf-8")


def _extract_from_image_bytes(file_bytes: bytes, mime_type: str) -> str:
    """
    Send an image (PNG/JPEG) to Gemini Vision and return extracted text.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            types.Part.from_text(
                text=(
                    "Extract all text from this image exactly as it appears. "
                    "Preserve line breaks and formatting where possible. "
                    "Output only the extracted text, nothing else."
                )
            ),
        ],
    )
    return response.text.strip()


def _extract_from_scanned_pdf(file_bytes: bytes) -> str:
    """
    Handle scanned PDFs by sending each page to Gemini Vision via the
    Gemini Files API (handles multi-page documents gracefully).
    """
    import pypdf
    import io

    try:
        from PIL import Image
        import pypdf
    except ImportError as e:
        raise ImportError(
            "Pillow and pypdf are required for scanned PDF OCR. "
            "Install with: pip install Pillow pypdf"
        ) from e

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    extracted_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        # First try native text extraction (fast path for digital PDFs)
        native_text = page.extract_text() or ""
        if native_text.strip():
            extracted_pages.append(native_text.strip())
            continue

        # Scanned page — render to image and OCR with Gemini
        if "/XObject" in (page.resources or {}):
            xobjects = page.resources["/XObject"].get_object()
            for obj_name, obj_ref in xobjects.items():
                obj = obj_ref.get_object()
                if obj.get("/Subtype") == "/Image":
                    image_data = obj.get_data()
                    filter_type = obj.get("/Filter", "")
                    if filter_type == "/DCTDecode":
                        mime = "image/jpeg"
                    else:
                        # Convert to PNG via Pillow as a safe fallback
                        try:
                            width = obj["/Width"]
                            height = obj["/Height"]
                            img = Image.frombytes("RGB", (width, height), image_data)
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            image_data = buf.getvalue()
                            mime = "image/png"
                        except Exception:
                            continue

                    page_text = _extract_from_image_bytes(image_data, mime)
                    extracted_pages.append(f"[Page {page_num}]\n{page_text}")
                    break
        else:
            extracted_pages.append(f"[Page {page_num}: no extractable content]")

    return "\n\n".join(extracted_pages)


def extract_text_ocr(source, mime_type: str | None = None) -> str:
    """
    Main OCR entry point. Accepts:
      - A file path (str or Path)
      - Raw bytes + explicit mime_type
      - A Streamlit UploadedFile object

    Supported formats: image/png, image/jpeg, application/pdf

    Returns the extracted text as a string.
    """
    SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "application/pdf"}

    # ── Streamlit UploadedFile ──────────────────────────────────────────────
    if hasattr(source, "read") and hasattr(source, "type"):
        file_bytes = source.read()
        mime_type = source.type
        source.seek(0)  # reset so callers can re-read if needed

    # ── File path ───────────────────────────────────────────────────────────
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        file_bytes = path.read_bytes()
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(path))

    # ── Raw bytes ───────────────────────────────────────────────────────────
    elif isinstance(source, bytes):
        file_bytes = source
        if mime_type is None:
            raise ValueError("mime_type must be provided when passing raw bytes.")

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported MIME type '{mime_type}'. "
            f"Supported: {SUPPORTED_MIME_TYPES}"
        )

    if mime_type == "application/pdf":
        return _extract_from_scanned_pdf(file_bytes)
    else:
        return _extract_from_image_bytes(file_bytes, mime_type)
