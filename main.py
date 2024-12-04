import streamlit as st
import numpy as np
import wave
import io
import speech_recognition as sr
from streamlit_advanced_audio import audix, WaveSurferOptions

# Title of the app
st.title("🎤 Interactive Audio Recorder and Transcriber")

# Record audio using st.audio_input
audio_data = st.audio_input("Please record your message:")

if audio_data:
    # Display the advanced audio player with waveform visualization
    audio_bytes = audio_data.read()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    options = WaveSurferOptions(
        wave_color="#2B88D9",
        progress_color="#b91d47",
        height=100
    )
    audix("temp_audio.wav", wavesurfer_options=options)

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio)
            st.subheader("Transcription:")
            st.write(transcription)
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
