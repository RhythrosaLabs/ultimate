import streamlit as st
import openai
import tempfile

# Set up the app title
st.title("Voice Recorder and Transcription App")

# Input for OpenAI API key
if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="Enter your API key here...",
)

if api_key:
    st.session_state["api_key"] = api_key

# Validate API key
if not st.session_state["api_key"]:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    openai.api_key = st.session_state["api_key"]

    # Audio recording input
    audio_file = st.audio_input("Record your voice")

    if audio_file:
        # Display audio playback
        st.audio(audio_file, format="audio/wav")

        # Save the audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper
        try:
            with st.spinner("Transcribing..."):
                with open(temp_audio_path, "rb") as audio:
                    transcript = openai.Audio.transcribe("whisper-1", audio)
                st.success("Transcription completed!")
                st.write("**Transcription:**")
                st.write(transcript["text"])

                # Option to download the transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcript["text"],
                    file_name="transcription.txt",
                    mime="text/plain",
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
