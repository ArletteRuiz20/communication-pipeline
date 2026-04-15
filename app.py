import streamlit as st
import os
import base64
from dotenv import load_dotenv
from google import genai
from agents.communication_pipeline.agent import cartesia_text_to_speech
from ocr_utils import extract_text_ocr

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def ask_gemini(prompt):
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text


def process_prompt(user_prompt, text_content=""):
    combined = f"{user_prompt}\n\n{text_content}".strip() if text_content else user_prompt

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

    import json, re
    try:
        clean = re.sub(r"```.*?```", "", detection, flags=re.DOTALL).strip()
        info = json.loads(clean)
    except Exception:
        info = {"task": "general", "language": "English", "language_code": "en", "wants_audio": False}

    task = info.get("task", "general")
    language = info.get("language", "English")
    lang_code = info.get("language_code", "en")
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
        return "\n".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(file_bytes)).pages)
    elif "wordprocessingml" in file_type:
        import docx, io
        return "\n".join(p.text for p in docx.Document(io.BytesIO(file_bytes)).paragraphs)
    return ""


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Communication Assistant", page_icon="🌍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 620px; padding-top: 0 !important; padding-bottom: 3rem; }

.hero { text-align: center; padding: 64px 0 24px; }
.hero-icon { font-size: 1.9rem; display: block; margin-bottom: 12px; }
.hero-title { font-size: 1.75rem; font-weight: 600; color: #111; letter-spacing: -0.03em; margin: 0 0 24px; }

div[data-testid="stButton"] button {
    background: #f5f5f5 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 20px !important;
    color: #555 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    padding: 4px 12px !important;
    height: auto !important;
    min-height: unset !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    line-height: 1.6 !important;
}
div[data-testid="stButton"] button:hover {
    background: #eaeaea !important; border-color: #ccc !important; color: #111 !important;
}

div[data-testid="stTextInput"] input {
    border-radius: 12px !important;
    border: 1.5px solid #e0e0e0 !important;
    background: #f7f7f5 !important;
    font-size: 0.95rem !important;
    color: #111 !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #999 !important; background: #fff !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important; outline: none !important;
}
div[data-testid="stTextInput"] label { display: none !important; }

.attach-btn {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f5f5f5; border: 1px solid #e8e8e8; border-radius: 20px;
    color: #555; font-size: 0.75rem; padding: 4px 12px;
    cursor: pointer; transition: background .15s;
    font-family: 'Inter', sans-serif;
    text-decoration: none;
}
.attach-btn:hover { background: #eaeaea; border-color: #ccc; color: #111; }

.attach-row { display: flex; justify-content: center; gap: 10px; margin-top: 8px; }

.badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #f0f0ee; border: 1px solid #ddd; border-radius: 20px;
    padding: 2px 9px; font-size: 0.72rem; color: #555; margin: 4px 3px 0 0;
}

.rlabel { font-size: 0.67rem; font-weight: 600; letter-spacing:.1em; text-transform:uppercase; color:#bbb; margin:22px 0 7px; }
.rcard { background:#fafaf8; border:1px solid #e8e8e6; border-radius:14px; padding:1.1rem 1.3rem; font-size:.93rem; line-height:1.75; color:#222; white-space:pre-wrap; }

/* Submit button */
.submit-wrap { display: flex; justify-content: center; margin-top: 14px; }
.submit-wrap > div > button {
    background: #111 !important; color: #fff !important;
    border-color: #111 !important; border-radius: 20px !important;
    padding: 6px 24px !important; font-size: 0.82rem !important;
}
.submit-wrap > div > button:hover { background: #333 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("doc_bytes", None), ("doc_name", None), ("doc_type", None),
             ("ocr_bytes", None), ("ocr_name", None), ("ocr_type", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-icon">🌍</span>
  <h1 class="hero-title">AI Communication Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ── Suggestion chips ──────────────────────────────────────────────────────────
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1: st.button("🌐 Translate text",    key="chip1")
with r1c2: st.button("📝 Summarize text",    key="chip2")
with r1c3: st.button("✨ Simplify language", key="chip3")
_, r2c1, _, r2c2, _ = st.columns([0.5, 2, 0.2, 2, 0.5])
with r2c1: st.button("🎭 Explain the tone",  key="chip4")
with r2c2: st.button("🔊 Translate + Audio", key="chip5")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Prompt bar ────────────────────────────────────────────────────────────────
prompt = st.text_input(
    "prompt",
    placeholder='e.g. "Translate this to Spanish and give me audio"',
    label_visibility="collapsed",
    key="prompt_bar"
)

# ── Attach buttons — HTML labels wired to hidden file inputs ──────────────────
# We use a Streamlit file_uploader hidden behind a styled HTML label
# so clicking the label opens the native OS file picker directly.

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# Render two real st.file_uploaders but visually hide them,
# then overlay styled labels on top via HTML.
doc_col, ocr_col = st.columns(2)

with doc_col:
    doc_file = st.file_uploader(
        "📎 Attach document",
        type=["txt", "pdf", "docx"],
        key="doc_uploader",
        label_visibility="visible"   # keep label visible — it IS the button
    )
    if doc_file:
        st.session_state.doc_bytes = doc_file.read()
        st.session_state.doc_name  = doc_file.name
        st.session_state.doc_type  = doc_file.type

with ocr_col:
    ocr_file = st.file_uploader(
        "📷 Attach image / OCR",
        type=["png", "jpg", "jpeg", "pdf"],
        key="ocr_uploader",
        label_visibility="visible"
    )
    if ocr_file:
        st.session_state.ocr_bytes = ocr_file.read()
        st.session_state.ocr_name  = ocr_file.name
        st.session_state.ocr_type  = ocr_file.type

# Hide ONLY the dropzone box, keep the label (which acts as the button)
st.markdown("""
<style>
[data-testid="stFileUploaderDropzone"] { display: none !important; }
section[data-testid="stFileUploader"] > label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    background: #f5f5f5 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 20px !important;
    color: #555 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    padding: 4px 12px !important;
    cursor: pointer !important;
    transition: background .15s !important;
    margin-bottom: 0 !important;
    width: auto !important;
}
section[data-testid="stFileUploader"] > label:hover {
    background: #eaeaea !important; border-color: #ccc !important; color: #111 !important;
}
section[data-testid="stFileUploader"] {
    display: flex !important;
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)

# ── Attachment badges ─────────────────────────────────────────────────────────
has_doc = st.session_state.doc_bytes is not None
has_ocr = st.session_state.ocr_bytes is not None

if has_doc or has_ocr:
    b = ""
    if has_doc: b += f'<span class="badge">📎 {st.session_state.doc_name}</span>'
    if has_ocr: b += f'<span class="badge">📷 {st.session_state.ocr_name}</span>'
    st.markdown(f"<div style='text-align:center;margin-top:6px'>{b}</div>", unsafe_allow_html=True)
    _, clr, _ = st.columns([2, 1, 2])
    with clr:
        if st.button("✕ Clear", key="clear"):
            for k in ["doc_bytes","doc_name","doc_type","ocr_bytes","ocr_name","ocr_type"]:
                st.session_state[k] = None
            st.rerun()

# ── Submit ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
_, scol, _ = st.columns([2, 1, 2])
with scol:
    submit = st.button("Submit →", key="submit")

# ── Process ───────────────────────────────────────────────────────────────────
if submit:
    text_content = ""

    if st.session_state.ocr_bytes:
        with st.spinner("Running OCR…"):
            try:
                text_content = extract_text_ocr(
                    st.session_state.ocr_bytes,
                    mime_type=st.session_state.ocr_type
                )
                st.markdown('<div class="rlabel">OCR Extracted Text</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="rcard">{text_content}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()
    elif st.session_state.doc_bytes:
        text_content = read_uploaded_bytes(
            st.session_state.doc_bytes,
            st.session_state.doc_name,
            st.session_state.doc_type
        )

    if not prompt.strip() and not text_content.strip():
        st.warning("Please type a message or attach a file.")
        st.stop()

    with st.spinner("Processing…"):
        result, wants_audio, lang_code = process_prompt(prompt, text_content)

    st.markdown('<div class="rlabel">Result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rcard">{result}</div>', unsafe_allow_html=True)

    if wants_audio:
        with st.spinner("Generating audio…"):
            try:
                cartesia_text_to_speech(result, lang_code)
                st.markdown('<div class="rlabel">Audio</div>', unsafe_allow_html=True)
                if os.path.exists("cartesia_output.wav"):
                    st.audio("cartesia_output.wav", format="audio/wav")
                    st.success("Audio generated!")
            except Exception as e:
                st.error(f"Audio generation failed: {e}")