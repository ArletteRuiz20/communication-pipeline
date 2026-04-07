# Communication Pipeline 🌐🔊

> ⚠️ Work in progress — this project is actively being developed.

## What it does
A multi-agent AI pipeline that:
- Translates text from any language
- Summarizes the translated text
- Converts the summary into audio

## Built with
- [Google ADK](https://google.github.io/adk-docs/) — multi-agent orchestration
- [Cartesia](https://cartesia.ai) — text to speech
- [Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) — translation & summarization

## Status
- [x] Translation agent
- [x] Summarization agent
- [x] Audio generation agent
- [ ] Web UI
- [ ] Support for more voices
- [ ] File input support

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```
3. Add your API keys to a `.env` file
4. Run the pipeline
```bash
   python main.py
```
