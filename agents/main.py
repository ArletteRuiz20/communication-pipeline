import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import InMemoryRunner
from google.genai import types
from agents.communication_pipeline.agent import root_agent


async def main():
    runner = InMemoryRunner(agent=root_agent)

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="user1"
    )

    user_language = input("What language do you want the output in? (e.g. Spanish, French, Japanese): ")
    user_input = input("Enter text in any language: ")
    user_input = f"Translate and summarize the following text. Output everything in {user_language}. Text: {user_input}"

    message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )

    print("Running pipeline...\n")

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
