import streamlit as st
import requests
import tempfile
from transformers import pipeline
from langdetect import detect

# App title
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

    persona_prompts = {
        "Shakespearean Poet": "Respond with a rhyming couplet in the style of Shakespeare.",
        "Modern Rapper": "Drop a rap line with a rhyme and rhythm based on the user's input.",
        "Playful Jokester": "Respond with a rhyming line that's humorous and playful."
    }

    system_prompt = persona_prompts[personality]

    # Record audio using Streamlit's audio_input widget
    audio_file = st.audio_input("Say something, and I'll rhyme!")

    if audio_file:
        # Display the recorded audio for playback
        st.audio(audio_file, format="audio/wav")

        # Save the audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper API
        try:
            with st.spinner("Transcribing your voice..."):
                with open(temp_audio_path, "rb") as audio:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                    }
                    files = {
                        "file": audio,
                    }
                    data = {
                        "model": "whisper-1",
                    }
                    response = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                    )
                if response.status_code == 200:
                    transcription = response.json()["text"]
                    st.success("Transcription completed!")
                    st.write("**You said:**")
                    st.write(transcription)

                    # Language Detection
                    language = detect(transcription)
                    if language != "en":
                        system_prompt = f"Respond with a rhyme in {language}."

                    # Sentiment Analysis
                    sentiment_analyzer = pipeline("sentiment-analysis")
                    sentiment = sentiment_analyzer(transcription)[0]["label"]

                    if sentiment == "NEGATIVE":
                        tone = "uplifting and encouraging"
                    elif sentiment == "POSITIVE":
                        tone = "joyful and celebratory"
                    else:
                        tone = "neutral and calm"

                    system_prompt += f" Respond in a tone that is {tone}."

                    # Generate a rhyming response using GPT Chat API
                    with st.spinner("Let me rhyme..."):
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        }
                        chat_data = {
                            "model": "gpt-4",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": system_prompt,
                                },
                                {
                                    "role": "user",
                                    "content": transcription,
                                },
                            ],
                            "temperature": 0.8,
                        }
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=chat_data,
                        )
                        if response.status_code == 200:
                            rhyming_line = response.json()["choices"][0]["message"]["content"].strip()
                            st.write("**Rhyme Bot says:**")
                            st.write(rhyming_line)
                        else:
                            st.error(
                                f"Failed to generate rhyme: {response.status_code} {response.text}"
                            )
                else:
                    st.error(
                        f"Failed to transcribe audio: {response.status_code} {response.text}"
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
