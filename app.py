import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from agents.communication_pipeline.agent import cartesia_text_to_speech

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def run_full_pipeline(text, language):
    translated = ask_gemini(
        f"Translate the following text into {language}. Output only the translated text:\n\n{text}"
    )
    audio_result = cartesia_text_to_speech(translated)
    return translated, audio_result

def run_single_task(text, task, language):
    if task == "Translate":
        prompt = f"Translate the following text into {language}. Output only the translated text:\n\n{text}"
    elif task == "Summarize":
        prompt = f"Summarize the following text clearly in {language}:\n\n{text}"
    elif task == "Simplify":
        prompt = f"Simplify the following text for a non-native speaker. Output in {language}:\n\n{text}"
    elif task == "Explain Tone":
        prompt = f"Analyze the tone of this message and explain it in {language}:\n\n{text}"
    return ask_gemini(prompt)

def read_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

st.set_page_config(page_title="AI Communication Assistant", page_icon="🌍")
st.title("🌍 Agentic AI Communication Assistant")
st.write("Enter text below or upload a file and choose a task.")

task = st.selectbox(
    "Select a task",
    ["Translate", "Summarize", "Simplify", "Explain Tone", "Full Pipeline (Translate + Audio)"]
)

language = st.selectbox(
    "Output language",
    ["Spanish", "French", "Arabic", "Chinese", "Japanese", "English"]
)

uploaded_file = st.file_uploader(
    "Upload a file (optional)",
    type=["txt", "pdf", "docx"]
)

user_input = st.text_area(
    "Or type your text here",
    height=200,
    placeholder="Paste a message, email, or document text..."
)

if st.button("Submit"):
    if uploaded_file is not None:
        user_input = read_file(uploaded_file)

    if user_input.strip() == "":
        st.warning("Please enter some text or upload a file.")
    elif task == "Full Pipeline (Translate + Audio)":
        with st.spinner("Translating and generating audio..."):
            translated, audio_result = run_full_pipeline(user_input, language)
        st.subheader("1️⃣ Translation")
        st.write(translated)
        st.subheader("2️⃣ Audio")
        if os.path.exists("cartesia_output.wav"):
            st.audio("cartesia_output.wav", format="audio/wav")
            st.success("Audio generated!")
        else:
            st.error(audio_result)
    else:
        with st.spinner("Processing..."):
            result = run_single_task(user_input, task, language)
        st.subheader("Result")
        st.write(result)