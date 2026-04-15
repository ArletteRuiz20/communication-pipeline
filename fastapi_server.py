import os
import io
import uuid
import json
import re
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

from google import genai
from agents.communication_pipeline.agent import cartesia_text_to_speech
from ocr_utils import extract_text_ocr

# Gmail OAuth
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
import base64
import email as email_lib

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ---------------------------------------------------------------------------
# Gmail OAuth config
# ---------------------------------------------------------------------------

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE       = "token.json"
REDIRECT_URI     = "http://localhost:8000/oauth/callback"


def get_gmail_service():
    """Return an authenticated Gmail service, refreshing token if needed."""
    if not os.path.exists(TOKEN_FILE):
        return None

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, GMAIL_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    if creds and creds.valid:
        return build("gmail", "v1", credentials=creds)

    return None


# ---------------------------------------------------------------------------
# Serve UI
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse("static/assistant-ui.html")


# ---------------------------------------------------------------------------
# Gmail OAuth endpoints
# ---------------------------------------------------------------------------

# Store active flows by state so callback can reuse the same flow object
_oauth_flows: dict = {}


@app.get("/oauth/login")
def oauth_login():
    """Start the Gmail OAuth flow — redirects user to Google login."""
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=GMAIL_SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # Save flow so callback can fetch the token with the same session
    _oauth_flows[state] = flow
    return RedirectResponse(auth_url)


@app.get("/oauth/callback")
def oauth_callback(code: str, state: str = ""):
    """Google redirects here after login — saves the token and returns to UI."""
    # Retrieve the original flow that started this OAuth session
    flow = _oauth_flows.pop(state, None)

    if not flow:
        # Fallback: create a new flow (works when state was lost between restarts)
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_FILE,
            scopes=GMAIL_SCOPES,
            redirect_uri=REDIRECT_URI,
        )

    # Tell oauthlib it's okay if scopes changed slightly
    import os as _os
    _os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

    flow.fetch_token(code=code)
    creds = flow.credentials
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return RedirectResponse("/?gmail=connected")


@app.get("/gmail/status")
def gmail_status():
    """Check if Gmail is connected."""
    service = get_gmail_service()
    if service:
        profile = service.users().getProfile(userId="me").execute()
        return {"connected": True, "email": profile.get("emailAddress", "")}
    return {"connected": False, "email": ""}


@app.get("/gmail/disconnect")
def gmail_disconnect():
    """Remove saved token."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
    return {"disconnected": True}


# ---------------------------------------------------------------------------
# Gmail endpoints
# ---------------------------------------------------------------------------

@app.get("/gmail/inbox")
def gmail_inbox(max: int = 10):
    """Fetch the latest emails from inbox."""
    service = get_gmail_service()
    if not service:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    msgs = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max,
    ).execute().get("messages", [])

    emails = []
    for m in msgs:
        msg = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()

        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        emails.append({
            "id":      m["id"],
            "from":    headers.get("From", ""),
            "subject": headers.get("Subject", "(no subject)"),
            "date":    headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
        })

    return {"emails": emails}


@app.get("/gmail/email/{email_id}")
def gmail_get_email(email_id: str):
    """Get the full body of a single email."""
    service = get_gmail_service()
    if not service:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    msg = service.users().messages().get(
        userId="me", id=email_id, format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    body = ""

    def extract_body(payload):
        if payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for part in payload.get("parts", []):
            result = extract_body(part)
            if result:
                return result
        return ""

    body = extract_body(msg["payload"])

    return {
        "id":      email_id,
        "from":    headers.get("From", ""),
        "to":      headers.get("To", ""),
        "subject": headers.get("Subject", ""),
        "date":    headers.get("Date", ""),
        "body":    body,
    }


@app.post("/gmail/process")
async def gmail_process(
    email_id: str = Form(...),
    action:   str = Form(...),   # "translate", "summarize", "draft_reply"
    prompt:   str = Form(""),
):
    """Run Gemini on an email — translate, summarize, or draft a reply."""
    service = get_gmail_service()
    if not service:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    # Fetch the email body
    email_data = gmail_get_email(email_id)
    body  = email_data.get("body", "")
    subj  = email_data.get("subject", "")
    frm   = email_data.get("from", "")
    email_text = f"Subject: {subj}\nFrom: {frm}\n\n{body}"

    if action == "translate":
        result, wants_audio, lang_code = run_task(
            prompt or "Translate this email to English", email_text
        )
    elif action == "summarize":
        result, wants_audio, lang_code = run_task(
            prompt or "Summarize this email clearly", email_text
        )
    elif action == "draft_reply":
        instruction = prompt or "Draft a professional reply to this email"
        result = ask_gemini(
            f"{instruction}\n\nOriginal email:\n{email_text}"
        )
        wants_audio = False
        lang_code = "en"
    else:
        result, wants_audio, lang_code = run_task(prompt, email_text)

    audio_url = None
    if wants_audio:
        try:
            cartesia_text_to_speech(result, lang_code)
            if os.path.exists("cartesia_output.wav"):
                filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
                dest = os.path.join("static", "audio", filename)
                shutil.move("cartesia_output.wav", dest)
                audio_url = f"/static/audio/{filename}"
        except Exception as e:
            print(f"Audio generation failed: {e}")

    return {
        "result":    result,
        "audio_url": audio_url,
        "subject":   subj,
        "from":      frm,
    }


@app.post("/gmail/send")
async def gmail_send(
    to:      str = Form(...),
    subject: str = Form(...),
    body:    str = Form(...),
):
    """Send an email."""
    service = get_gmail_service()
    if not service:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    msg = email_lib.message.EmailMessage()
    msg["To"]      = to
    msg["Subject"] = subject
    msg.set_content(body)

    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(
        userId="me", body={"raw": encoded}
    ).execute()

    return {"sent": True}


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def ask_gemini(prompt: str) -> str:
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    ).text


def detect_intent(user_prompt: str, text_content: str = "") -> dict:
    combined = f"{user_prompt}\n\n{text_content}".strip() if text_content else user_prompt
    raw = ask_gemini(f"""
Analyze this user request and respond in JSON only (no markdown, no backticks):
{{
  "task": "translate | summarize | simplify | explain_tone | general",
  "language": "the target language name in English, e.g. Spanish, French, Japanese. Use English if not specified.",
  "language_code": "ISO 639-1 code e.g. es, fr, ja, en",
  "wants_audio": true or false (true if user mentions audio, voice, speak, listen, sound, or asks for translation)
}}

User request: {combined}
""")
    try:
        clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
        return json.loads(clean)
    except Exception:
        return {"task": "general", "language": "English", "language_code": "en", "wants_audio": False}


def run_task(user_prompt: str, text_content: str = "") -> tuple[str, bool, str]:
    combined = f"{user_prompt}\n\n{text_content}".strip() if text_content else user_prompt
    info      = detect_intent(user_prompt, text_content)
    task        = info.get("task", "general")
    language    = info.get("language", "English")
    lang_code   = info.get("language_code", "en")
    wants_audio = info.get("wants_audio", False)

    if task == "translate":
        result = ask_gemini(f"Translate the following into {language}. Output only the translation:\n\n{combined}")
    elif task == "summarize":
        result = ask_gemini(f"Summarize the following text clearly in {language}:\n\n{combined}")
    elif task == "simplify":
        result = ask_gemini(f"Simplify the following text for a non-native speaker. Output in {language}:\n\n{combined}")
    elif task == "explain_tone":
        result = ask_gemini(f"Analyze the tone of this message and explain it in {language}:\n\n{combined}")
    else:
        result = ask_gemini(combined)

    return result, wants_audio, lang_code


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_document(content: bytes, content_type: str) -> str:
    if "text/plain" in content_type:
        return content.decode("utf-8", errors="ignore")
    if "pdf" in content_type:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if "wordprocessingml" in content_type or "msword" in content_type:
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


# ---------------------------------------------------------------------------
# /run endpoint (unchanged)
# ---------------------------------------------------------------------------

@app.post("/run")
async def run_endpoint(
    prompt:   str = Form(...),
    doc_file: Optional[UploadFile] = File(None),
    ocr_file: Optional[UploadFile] = File(None),
):
    text_content = ""
    ocr_text     = ""

    if ocr_file and ocr_file.filename:
        raw      = await ocr_file.read()
        ocr_text = extract_text_ocr(raw, mime_type=ocr_file.content_type or "")
        text_content = ocr_text
    elif doc_file and doc_file.filename:
        raw          = await doc_file.read()
        text_content = read_document(raw, doc_file.content_type or "")

    result, wants_audio, lang_code = run_task(prompt, text_content)

    audio_url = None
    if wants_audio:
        try:
            cartesia_text_to_speech(result, lang_code)
            if os.path.exists("cartesia_output.wav"):
                filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
                dest = os.path.join("static", "audio", filename)
                shutil.move("cartesia_output.wav", dest)
                audio_url = f"/static/audio/{filename}"
        except Exception as e:
            print(f"Audio generation failed: {e}")

    return {
        "result":    result,
        "ocr_text":  ocr_text if ocr_text else None,
        "audio_url": audio_url,
    }