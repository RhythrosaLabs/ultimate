import streamlit as st
import requests
import tempfile

# Streamlit app title
st.title("Voice Recorder and Transcription App")

# Input for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="Enter your API key here...",
)

if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    # Record audio using Streamlit's audio_input widget
    audio_file = st.audio_input("Record your voice")

    if audio_file:
        # Display the recorded audio for playback
        st.audio(audio_file, format="audio/wav")

        # Save the audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper API
        try:
            with st.spinner("Transcribing..."):
                with open(temp_audio_path, "rb") as audio:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                    }
                    files = {
                        "file": audio,
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
                    st.success("Transcription completed!")
                    st.write("**Transcription:**")
                    st.write(transcript["text"])

                    # Provide a download option for the transcription
                    st.download_button(
                        label="Download Transcription",
                        data=transcript["text"],
                        file_name="transcription.txt",
                        mime="text/plain",
                    )
                else:
                    st.error(
                        f"Failed to transcribe audio: {response.status_code} {response.text}"
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
