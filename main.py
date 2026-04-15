import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import InMemoryRunner
from google.genai import types
from agents.communication_pipeline.agent import root_agent
from ocr_utils import extract_text_ocr

async def main():
    runner = InMemoryRunner(agent=root_agent)

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1"
    )

    user_language = input("What language do you want the output in? (e.g. French, Japanese, Spanish): ").strip()
 
    # ── Input mode selection ─────────────────────────────────────────────────
    print("\nInput options:")
    print("  1. Type / paste text directly")
    print("  2. Provide a file path to an image or scanned PDF for OCR")
    mode = input("Choose (1 or 2): ").strip()
 
    if mode == "2":
        file_path = input("Enter the file path (PNG, JPEG, or PDF): ").strip()
        path = Path(file_path)
        if not path.exists():
            print(f"Error: file not found at '{file_path}'")
            return
        print(f"\nRunning OCR on '{path.name}'...")
        try:
            user_text = extract_text_ocr(path)
            print("\n── OCR Extracted Text ──────────────────────────────────")
            print(user_text)
            print("────────────────────────────────────────────────────────\n")
        except Exception as e:
            print(f"OCR failed: {e}")
            return
    else:
        user_text = input("Enter text in any language: ").strip()
 
    user_input = (
        f"Translate and summarize the following text. "
        f"Output everything in {user_language}. Text: {user_text}"
    )
 
    message = types.Content(
        role="user",
        parts=[types.Part(text=user_input)]
    )
 
    print("\nRunning pipeline...\n")
 
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=message
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"[{event.author}]: {part.text}")
 
    print("\nDone! Check for cartesia_output.wav in your project folder.")
 
 
if __name__ == "__main__":
    asyncio.run(main())