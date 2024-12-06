import streamlit as st
import requests
import tempfile
import json
from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import detect_nonsilent
from random import choice
import numpy as np

# Streamlit app title
st.title("Beat Slicing App")

# Input for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="Enter your API key here...",
)

def process_audio(api_key, audio_file):
    # Save the audio file temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # Ensure audio is properly encoded before sending to the API
    processed_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio = AudioSegment.from_file(temp_audio_path)
    audio.export(processed_audio_path, format="wav")

    # Transcribe audio using OpenAI Whisper API
    try:
        with st.spinner("Transcribing and slicing the audio..."):
            with open(processed_audio_path, "rb") as audio_file:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }
                files = {
                    "file": audio_file,
                }
                data = {
                    "model": "whisper-1",  # Specify the Whisper model
                }
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                )

            if response.status_code == 200:
                transcript = response.json()
                st.write("API Response:", transcript)  # Debugging output

                if "segments" in transcript:
                    words = transcript["segments"]

                    # Load the audio file into pydub for slicing
                    audio = AudioSegment.from_file(temp_audio_path)
                    word_slices = []

                    # Slice the audio based on word timestamps
                    for segment in words:
                        start_ms = int(segment["start"] * 1000)
                        end_ms = int(segment["end"] * 1000)
                        word_audio = audio[start_ms:end_ms]
                        word_slices.append(word_audio)

                elif "text" in transcript:
                    st.warning("Word-level timestamps are missing. Estimating word positions.")

                    # Estimate word durations
                    words = transcript["text"].split()
                    num_words = len(words)
                    total_duration = len(audio)
                    word_duration = total_duration // num_words

                    # Slice the audio into estimated word chunks
                    word_slices = []
                    for i, word in enumerate(words):
                        start_ms = i * word_duration
                        end_ms = start_ms + word_duration
                        word_audio = audio[start_ms:end_ms]
                        word_slices.append(word_audio)

                else:
                    st.error("The transcription response did not contain segments or text. Please try again.")
                    return

                # Create a beat loop using samples
                beat_loop = AudioSegment.silent(duration=4000)  # 4 seconds base loop
                for i in range(16):  # Add 16 samples
                    sample = choice(word_slices)
                    start_time = int((i / 16) * 4000)
                    beat_loop = beat_loop.overlay(sample, position=start_time)

                # Export the generated beat loop
                beat_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                beat_loop.export(beat_path, format="wav")

                # Display results
                st.success("Transcription and beat slicing completed!")
                st.write("**Generated Beat Loop:**")
                st.audio(beat_path, format="audio/wav")

                # Provide a download option for the beat loop
                st.download_button(
                    label="Download Beat Loop",
                    data=open(beat_path, "rb").read(),
                    file_name="beat_loop.wav",
                    mime="audio/wav",
                )
            else:
                st.error(
                    f"Failed to transcribe audio: {response.status_code} {response.text}"
                )
    except Exception as e:
        st.error(f"An error occurred: {e}")

if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    # Record audio using Streamlit's audio_input widget
    audio_file = st.audio_input("Record your voice")

    if audio_file:
        # Display the recorded/uploaded audio for playback
        st.audio(audio_file, format="audio/wav")

        # Process the audio
        process_audio(api_key, audio_file)
