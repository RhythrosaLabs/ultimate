import streamlit as st
import requests
import tempfile
from transformers import pipeline
from langdetect import detect
import json
from gradio_client import Client, handle_file

# Initialize Gradio Client
GRADIO_SERVER_URL = "http://localhost:7788/"
client = Client(GRADIO_SERVER_URL)

# --------------------- Helper Functions --------------------- #

def run_gradio_agent(task):
    """Run an agent using the Gradio API."""
    result = client.predict(
        agent_type="custom",
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_temperature=1,
        llm_base_url="",
        llm_api_key=api_key,  # Using user-provided API key
        use_own_browser=False,
        keep_browser_open=False,
        headless=False,
        disable_security=True,
        window_w=1280,
        window_h=1100,
        save_recording_path="./tmp/record_videos",
        save_agent_history_path="./tmp/agent_history",
        save_trace_path="./tmp/traces",
        enable_recording=True,
        task=task,
        add_infos="",
        max_steps=100,
        use_vision=True,
        max_actions_per_step=10,
        tool_calling_method="auto",
        api_name="/run_with_stream"
    )
    return result

# --------------------- Streamlit App --------------------- #

st.title("Rhyme Bot: Speak, and I'll Rhyme!")

# Input for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="Enter your API key here...",
)

if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    # Personality Selector
    personality = st.radio(
        "Choose a personality for Rhyme Bot:",
        ("Shakespearean Poet", "Modern Rapper", "Playful Jokester")
    )

    # Record audio
    audio_file = st.audio_input("Say something, and I'll rhyme!")

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper API
        try:
            with open(temp_audio_path, "rb") as audio:
                headers = {"Authorization": f"Bearer {api_key}"}
                files = {"file": audio}
                data = {"model": "whisper-1"}
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data
                )
            
            if response.status_code == 200:
                transcription = response.json()["text"]
                st.success("Transcription completed!")
                st.write("**You said:**", transcription)

                # Run Gradio Agent to process the task
                agent_response = run_gradio_agent(transcription)
                st.write("**Gradio Agent Response:**", agent_response)

            else:
                st.error(f"Failed to transcribe audio: {response.status_code} {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
