import os
from cartesia import Cartesia
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent


def cartesia_text_to_speech(text: str) -> str:
    """Converts text to speech using Cartesia and saves as WAV."""
    client = Cartesia(api_key=os.environ["CARTESIA_API_KEY"])

    generator = client.tts.bytes(
        model_id="sonic-english",
        transcript=text,
        voice={"mode": "id", "id": "a0e99841-438c-4a64-b679-ae501e7d6091"},
        output_format={
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    )

    file_path = "cartesia_output.wav"
    with open(file_path, "wb") as f:
        for chunk in generator:
            f.write(chunk)

    return f"Audio saved to {file_path}"


MODEL = "gemini-2.5-flash"

planner_agent = LlmAgent(
    name="planner_agent",
    model=MODEL,
    description="Plans the workflow for translation, summarization, and audio generation.",
    instruction="""
You are the planner agent.
Follow this workflow:
1. Translate the user's text into the language they requested
2. Summarize the translated text in 2-3 sentences in that same language
3. Convert the summary into audio

Output only a short one-line plan.
""",
    output_key="plan"
)

translator_agent = LlmAgent(
    name="translator_agent",
    model=MODEL,
    description="Translates the user's text into the requested language.",
    instruction="""
Translate the user's original text into the language specified in their request.
Preserve the meaning and tone as much as possible.
Output only the translated text in the requested language.
""",
    output_key="translated_text"
)

summarizer_agent = LlmAgent(
    name="summarizer_agent",
    model=MODEL,
    description="Summarizes translated text in the requested language.",
    instruction="""
Read the following translated text and summarize it in 2-3 concise sentences.
Keep the summary in the same language as the translated text.

Translated text:
{translated_text}

Output only the summary in the same language.
""",
    output_key="summary"
)

voice_agent = LlmAgent(
    name="voice_agent",
    model=MODEL,
    description="Converts summary into speech using Cartesia.",
    instruction="""
Take the following summary and call the cartesia_text_to_speech tool with it as input.
Pass the summary exactly as it is, in whatever language it is written.
Return only a short confirmation message after the audio is generated.

Summary:
{summary}
""",
    tools=[cartesia_text_to_speech],
    output_key="audio_result"
)

root_agent = SequentialAgent(
    name="translation_summary_audio_pipeline",
    description="Planner -> Translator -> Summarizer -> Voice",
    sub_agents=[planner_agent, translator_agent, summarizer_agent, voice_agent]
)
