import streamlit as st
import openai

# Initialize OpenAI API client
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Voice Recorder and Transcription App")

# Audio recording input
audio_file = st.audio_input("Record your message")

if audio_file:
    # Display audio playback
    st.audio(audio_file)

    # Transcribe audio using OpenAI Whisper
    with st.spinner("Transcribing..."):
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        st.write("**Transcription:**")
        st.write(transcript["text"])

        # Download transcription
        st.download_button(
            label="Download Transcription",
            data=transcript["text"],
            file_name="transcription.txt",
            mime="text/plain",
        )
