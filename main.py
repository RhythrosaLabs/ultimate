import streamlit as st
import requests
import tempfile

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

                    # Generate a rhyming response using GPT
                    with st.spinner("Let me rhyme..."):
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        }
                        prompt = (
                            f"You are a poet bot. The user said: '{transcription}'. "
                            f"Respond with a single line that rhymes with it and fits the theme."
                        )
                        data = {
                            "model": "gpt-4",
                            "prompt": prompt,
                            "temperature": 0.8,
                            "max_tokens": 50,
                        }
                        response = requests.post(
                            "https://api.openai.com/v1/completions",
                            headers=headers,
                            json=data,
                        )
                        if response.status_code == 200:
                            rhyming_line = response.json()["choices"][0]["text"].strip()
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
