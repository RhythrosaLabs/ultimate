import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import io
import speech_recognition as sr
from annotated_text import annotated_text

# Title of the app
st.title("🎤 Interactive Audio Recorder and Transcriber")

# Record audio using st.audio_input
audio_data = st.audio_input("Please record your message:")

if audio_data:
    # Display the audio player
    st.audio(audio_data)

    # Read audio data
    audio_bytes = audio_data.read()

    # Save the audio to a temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    # Visualize the audio waveform
    with wave.open("temp_audio.wav", "rb") as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
        framerate = wav_file.getframerate()

        # Time axis
        time = np.linspace(
            0, len(signal) / framerate, num=len(signal)
        )

        # Plot
        fig, ax = plt.subplots()
        ax.plot(time, signal)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        st.pyplot(fig)

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio)
            st.subheader("Transcription:")
            annotated_text(
                ("", transcription, "#8ef")
            )
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
