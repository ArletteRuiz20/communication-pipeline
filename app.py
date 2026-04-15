import streamlit as st
import os
import json
import base64
import email as email_lib
from dotenv import load_dotenv
from google import genai
from agents.communication_pipeline.agent import cartesia_text_to_speech
from ocr_utils import extract_text_ocr
from streamlit_oauth import OAuth2Component

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ---------------------------------------------------------------------------
# OAuth credentials
# ---------------------------------------------------------------------------

with open("credentials.json") as f:
    _creds = json.load(f)

_web          = _creds.get("web") or _creds.get("installed")
CLIENT_ID     = _web["client_id"]
CLIENT_SECRET = _web["client_secret"]
AUTH_URI      = _web["auth_uri"]
TOKEN_URI     = _web["token_uri"]
REDIRECT_URI  = "http://localhost:8501/component/streamlit_oauth.authorize_button/index.html"
SCOPES        = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send"

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def ask_gemini(prompt):
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text


def process_prompt(user_prompt, text_content=""):
    combined = f"{user_prompt}\n\n{text_content}".strip() if text_content else user_prompt

    import re
    detection = ask_gemini(f"""
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
        clean = re.sub(r"```.*?```", "", detection, flags=re.DOTALL).strip()
        info = json.loads(clean)
    except Exception:
        info = {"task": "general", "language": "English", "language_code": "en", "wants_audio": False}

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


def read_uploaded_bytes(file_bytes, file_name, file_type):
    if file_type == "text/plain":
        return file_bytes.decode("utf-8")
    elif file_type == "application/pdf":
        import pypdf, io
        return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(file_bytes)).pages)
    elif "wordprocessingml" in file_type:
        import docx, io
        return "\n".join(p.text for p in docx.Document(io.BytesIO(file_bytes)).paragraphs)
    return ""


def show_result(result, wants_audio, lang_code):
    """Helper to display result text and optional audio."""
    st.markdown("**Result**")
    st.text_area("", value=result, height=380, disabled=True, label_visibility="collapsed", key=f"res_{hash(result)}")
    if wants_audio:
        with st.spinner("Generating audio…"):
            try:
                cartesia_text_to_speech(result, lang_code)
                if os.path.exists("cartesia_output.wav"):
                    st.audio("cartesia_output.wav", format="audio/wav")
                    st.success("Audio ready!")
            except Exception as e:
                st.error(f"Audio failed: {e}")

# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

def get_gmail_service(token: dict):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=token["access_token"],
        refresh_token=token.get("refresh_token"),
        token_uri=TOKEN_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES.split(),
    )
    return build("gmail", "v1", credentials=creds)


def fetch_inbox(service, max_results=10):
    msgs = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute().get("messages", [])
    emails = []
    for m in msgs:
        msg = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        h = {x["name"]: x["value"] for x in msg["payload"]["headers"]}
        emails.append({
            "id":      m["id"],
            "from":    h.get("From", ""),
            "subject": h.get("Subject", "(no subject)"),
            "date":    h.get("Date", ""),
            "snippet": msg.get("snippet", ""),
        })
    return emails


def fetch_email_body(service, email_id):
    msg = service.users().messages().get(
        userId="me", id=email_id, format="full"
    ).execute()
    h = {x["name"]: x["value"] for x in msg["payload"]["headers"]}

    def extract(payload):
        if payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for part in payload.get("parts", []):
            r = extract(part)
            if r: return r
        return ""

    return {
        "id":      email_id,
        "from":    h.get("From", ""),
        "to":      h.get("To", ""),
        "subject": h.get("Subject", ""),
        "date":    h.get("Date", ""),
        "body":    extract(msg["payload"]),
    }


def send_email(service, to, subject, body):
    msg = email_lib.message.EmailMessage()
    msg["To"]      = to
    msg["Subject"] = subject
    msg.set_content(body)
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": encoded}).execute()


# ---------------------------------------------------------------------------
# Page config & minimal CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Communication Assistant",
    page_icon="🌍",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 700px; padding-top: 2rem !important; padding-bottom: 4rem; }
h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif !important; }
/* Clean up default Streamlit button */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    border: 1px solid #ddd !important;
    background: #f7f7f5 !important;
    color: #222 !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    padding: 6px 18px !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button:hover {
    background: #efefed !important;
    border-color: #bbb !important;
    color: #000 !important;
}
/* Primary action buttons */
div[data-testid="stButton"].primary > button {
    background: #111 !important;
    color: #fff !important;
    border-color: #111 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for k, v in [
    ("doc_bytes", None), ("doc_name", None), ("doc_type", None),
    ("ocr_bytes", None), ("ocr_name", None), ("ocr_type", None),
    ("gmail_token", None), ("selected_email", None), ("draft_reply", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("<div style='text-align:center;padding:16px 0 8px'>", unsafe_allow_html=True)
st.markdown("# 🌍 AI Communication Assistant")
st.markdown("<p style='color:#888;font-size:0.95rem;margin-top:-8px'>Translate · Summarize · Simplify · Draft · Audio</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_assistant, tab_gmail = st.tabs(["🧠 Assistant", "📧 Gmail"])


# ===========================================================================
# ASSISTANT TAB
# ===========================================================================

with tab_assistant:

    st.markdown("#### What would you like to do?")

    # Suggestion chips as a clean row of columns
    c1, c2, c3 = st.columns(3)
    with c1: st.button("🌐 Translate",       key="chip1")
    with c2: st.button("📝 Summarize",        key="chip2")
    with c3: st.button("✨ Simplify",         key="chip3")
    c4, c5, c6 = st.columns(3)
    with c4: st.button("🎭 Explain tone",     key="chip4")
    with c5: st.button("🔊 Translate + Audio",key="chip5")
    with c6: st.button("💬 General",          key="chip6")

    st.markdown("")

    # Prompt input
    prompt = st.text_input(
        "Your instruction",
        placeholder='e.g. "Translate this to French and generate audio"',
        key="prompt_bar",
    )

    st.markdown("")

    # File uploaders — native Streamlit, no CSS hacks
    st.markdown("**Attach files (optional)**")
    col_doc, col_ocr = st.columns(2)

    with col_doc:
        doc_file = st.file_uploader(
            "📎 Document (TXT, PDF, DOCX)",
            type=["txt", "pdf", "docx"],
            key="doc_uploader",
        )
        if doc_file:
            st.session_state.doc_bytes = doc_file.read()
            st.session_state.doc_name  = doc_file.name
            st.session_state.doc_type  = doc_file.type

    with col_ocr:
        ocr_file = st.file_uploader(
            "📷 Image / OCR (PNG, JPG, PDF)",
            type=["png", "jpg", "jpeg", "pdf"],
            key="ocr_uploader",
        )
        if ocr_file:
            st.session_state.ocr_bytes = ocr_file.read()
            st.session_state.ocr_name  = ocr_file.name
            st.session_state.ocr_type  = ocr_file.type

    # Show what's attached
    if st.session_state.doc_bytes or st.session_state.ocr_bytes:
        attached = []
        if st.session_state.doc_bytes: attached.append(f"📎 {st.session_state.doc_name}")
        if st.session_state.ocr_bytes: attached.append(f"📷 {st.session_state.ocr_name}")
        st.caption("Attached: " + "  ·  ".join(attached))
        if st.button("✕ Clear attachments", key="clear"):
            for k in ["doc_bytes","doc_name","doc_type","ocr_bytes","ocr_name","ocr_type"]:
                st.session_state[k] = None
            st.rerun()

    st.markdown("")
    col_sub1, col_sub2, col_sub3 = st.columns([1, 1, 1])
    with col_sub2:
        submit = st.button("Submit →", key="submit", use_container_width=True)

    # Processing
    if submit:
        text_content = ""

        if st.session_state.ocr_bytes:
            with st.spinner("Running OCR…"):
                try:
                    text_content = extract_text_ocr(
                        st.session_state.ocr_bytes,
                        mime_type=st.session_state.ocr_type,
                    )
                    with st.expander("OCR extracted text", expanded=False):
                        st.text(text_content[:2000])
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    st.stop()

        elif st.session_state.doc_bytes:
            text_content = read_uploaded_bytes(
                st.session_state.doc_bytes,
                st.session_state.doc_name,
                st.session_state.doc_type,
            )

        if not prompt.strip() and not text_content.strip():
            st.warning("Please type a message or attach a file.")
            st.stop()

        with st.spinner("Processing…"):
            result, wants_audio, lang_code = process_prompt(prompt, text_content)

        st.markdown("---")
        st.markdown("**Result**")
        st.text_area("result_out", value=result, height=380,
                     disabled=True, label_visibility="collapsed")

        if wants_audio:
            with st.spinner("Generating audio…"):
                try:
                    cartesia_text_to_speech(result, lang_code)
                    st.markdown("**Audio**")
                    if os.path.exists("cartesia_output.wav"):
                        st.audio("cartesia_output.wav", format="audio/wav")
                        st.success("Audio ready!")
                except Exception as e:
                    st.error(f"Audio failed: {e}")


# ===========================================================================
# GMAIL TAB
# ===========================================================================

with tab_gmail:

    oauth = OAuth2Component(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        authorize_endpoint=AUTH_URI,
        token_endpoint=TOKEN_URI,
        refresh_token_endpoint=TOKEN_URI,
        revoke_token_endpoint=None,
    )

    # ── Not connected ─────────────────────────────────────────────────────────
    if not st.session_state.gmail_token:
        st.markdown("")
        st.markdown("#### Connect your Gmail")
        st.markdown("Sign in to read, translate, summarize, and draft replies to your emails.")
        st.markdown("")

        result = oauth.authorize_button(
            name="Sign in with Gmail",
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            key="gmail_oauth",
            extras_params={"access_type": "offline", "prompt": "consent"},
            use_container_width=False,
            pkce="S256",
        )
        if result and "token" in result:
            st.session_state.gmail_token = result["token"]
            st.rerun()

    # ── Connected ─────────────────────────────────────────────────────────────
    else:
        token = st.session_state.gmail_token

        try:
            service = get_gmail_service(token)
            profile = service.users().getProfile(userId="me").execute()
            email_addr = profile.get("emailAddress", "")
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.session_state.gmail_token = None
            st.rerun()

        # Status row
        col_s, col_d = st.columns([4, 1])
        with col_s:
            st.success(f"✓ Connected as **{email_addr}**")
        with col_d:
            if st.button("Disconnect", key="disc"):
                st.session_state.gmail_token = None
                st.session_state.selected_email = None
                st.rerun()

        st.markdown("---")
        st.markdown("#### Inbox")

        col_r, _ = st.columns([1, 5])
        with col_r:
            if st.button("↻ Refresh", key="refresh"):
                st.session_state.selected_email = None
                st.rerun()

        with st.spinner("Loading inbox…"):
            try:
                emails = fetch_inbox(service, max_results=10)
            except Exception as e:
                st.error(f"Could not load inbox: {e}")
                emails = []

        if not emails:
            st.info("No emails found.")
        else:
            for em in emails:
                date_short = em["date"][:16] if em["date"] else ""
                label = f"**{em['subject']}**  ·  {em['from']}  ·  {date_short}"
                if st.button(label, key=f"em_{em['id']}", use_container_width=True):
                    st.session_state.selected_email = em["id"]
                    st.session_state.draft_reply = ""
                    st.rerun()

        # ── Selected email ────────────────────────────────────────────────────
        if st.session_state.selected_email:
            st.markdown("---")
            with st.spinner("Loading email…"):
                try:
                    email_data = fetch_email_body(service, st.session_state.selected_email)
                except Exception as e:
                    st.error(f"Failed to load email: {e}")
                    st.stop()

            st.markdown(f"**Subject:** {email_data['subject']}")
            st.markdown(f"**From:** {email_data['from']}")
            st.markdown(f"**Date:** {email_data['date']}")

            with st.expander("View email body", expanded=False):
                st.text(email_data["body"][:3000] or "(no body)")

            email_text = f"Subject: {email_data['subject']}\nFrom: {email_data['from']}\n\n{email_data['body']}"

            st.markdown("---")
            st.markdown("#### What do you want to do with this email?")
            st.caption('Type any instruction — e.g. "Summarize this", "Translate to French", "Draft a reply in Spanish", "Explain the tone and generate audio"')

            email_prompt = st.text_input(
                "Your instruction",
                placeholder='e.g. "Translate to Spanish and generate audio"',
                key="g_prompt",
                label_visibility="collapsed",
            )

            col_run1, col_run2, col_run3 = st.columns([1, 1, 1])
            with col_run2:
                run_email = st.button("Run →", key="g_run", use_container_width=True)

            if run_email:
                if not email_prompt.strip():
                    st.warning("Please type an instruction above.")
                else:
                    # Detect if user wants a draft reply — handle separately so it
                    # populates the send form rather than just showing a text area
                    lower_p = email_prompt.lower()
                    is_draft = any(w in lower_p for w in ["draft", "reply", "respond", "write back", "answer"])

                    if is_draft:
                        with st.spinner("Drafting reply…"):
                            draft = ask_gemini(
                                f"{email_prompt}\n\nOriginal email:\n{email_text}"
                            )
                        st.session_state.draft_reply = draft
                        st.rerun()
                    else:
                        with st.spinner("Processing…"):
                            res, wa, lc = process_prompt(email_prompt, email_text)
                        st.markdown("**Result**")
                        st.text_area("email_res_out", value=res, height=380,
                                     disabled=True, label_visibility="collapsed")
                        if wa:
                            with st.spinner("Generating audio…"):
                                try:
                                    cartesia_text_to_speech(res, lc)
                                    if os.path.exists("cartesia_output.wav"):
                                        st.audio("cartesia_output.wav", format="audio/wav")
                                        st.success("Audio ready!")
                                except Exception as e:
                                    st.error(f"Audio failed: {e}")

            # ── Draft compose area ────────────────────────────────────────────
            if st.session_state.draft_reply:
                st.markdown("---")
                st.markdown("#### Draft reply")
                to_addr = st.text_input("To", value=email_data["from"], key="dt")
                subj    = st.text_input("Subject", value=f"Re: {email_data['subject']}", key="ds")
                body    = st.text_area("Body", value=st.session_state.draft_reply,
                                       height=200, key="db")

                sc1, sc2, _ = st.columns([1, 1, 2])
                with sc1:
                    if st.button("📤 Send email", key="send_btn", use_container_width=True):
                        with st.spinner("Sending…"):
                            try:
                                send_email(service, to_addr, subj, body)
                                st.success(f"Sent to {to_addr}!")
                                st.session_state.draft_reply = ""
                            except Exception as e:
                                st.error(f"Send failed: {e}")
                with sc2:
                    if st.button("📋 Copy text", key="copy_btn", use_container_width=True):
                        st.code(body, language=None)
                        st.caption("Select all the text above and copy.")