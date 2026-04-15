# 🌍 AI Communication Assistant

An agentic AI application that breaks down language barriers in communication. Built with Google's Agent Development Kit (ADK), Gemini 2.5, and Cartesia text-to-speech — it translates, summarizes, simplifies, and vocalizes text across languages, with native Gmail integration for real-world email workflows.

---

## What it does

Users type a natural language instruction — the app figures out what to do:

- **Translate** any text or document into a target language
- **Summarize** long content into clear, concise output
- **Simplify** complex language for non-native speakers
- **Explain tone** of a message and what it communicates
- **Generate audio** of the result using Cartesia TTS
- **Read and process Gmail** — summarize, translate, or draft replies to real emails
- **OCR** — extract text from scanned PDFs and images before processing

No dropdowns. No task selection. Just a prompt.

---

## Demo

| Feature | Description |
|---|---|
| Natural language routing | Gemini detects intent and language from the prompt automatically |
| Document upload | Supports TXT, PDF, DOCX |
| Image / OCR | Extracts text from photos and scanned documents |
| Audio output | Cartesia generates spoken audio of the result |
| Gmail integration | OAuth 2.0 — reads inbox, summarizes emails, drafts and sends replies |

---

## Tech stack

| Layer | Technology |
|---|---|
| AI agents | Google ADK + Gemini 2.5 Flash |
| Text to speech | Cartesia API |
| OCR | Tesseract + pytesseract |
| Web UI | Streamlit |
| Gmail | Google Gmail API + OAuth 2.0 |
| Backend | FastAPI + Uvicorn |
| Deployment | Streamlit Cloud |

---

## Architecture

```
User prompt
    ↓
Gemini 2.5 Flash — detects task, language, audio intent
    ↓
ADK agent pipeline
    ├── Translation agent
    ├── Summarization agent  
    ├── Simplification agent
    ├── Tone analysis agent
    └── Cartesia audio agent
    ↓
Streamlit UI — displays result + audio player
```

---

## Running locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/communication-pipeline.git
cd communication-pipeline

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Fill in GOOGLE_API_KEY and CARTESIA_API_KEY

# Run
streamlit run app.py
```

---

## Environment variables

```
GOOGLE_API_KEY=your_gemini_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

For Gmail, place your `credentials.json` (Google OAuth client) in the project root.

---

## Project structure

```
communication-pipeline/
├── agents/
│   └── communication_pipeline/
│       └── agent.py          # ADK agent definitions
├── static/
│   └── assistant-ui.html     # Custom web UI
├── app.py                    # Streamlit app
├── fastapi_server.py         # FastAPI server
├── ocr_utils.py              # OCR helper
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## Built with

- [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- [Google Gemini API](https://ai.google.dev/)
- [Cartesia TTS](https://cartesia.ai/)
- [Streamlit](https://streamlit.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Gmail API](https://developers.google.com/gmail/api)