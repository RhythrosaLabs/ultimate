import streamlit as st
import io
import speech_recognition as sr
from streamlit_advanced_audio import audix, WaveSurferOptions

# -------------------------
# App Configuration
# -------------------------

st.set_page_config(
    page_title="🎤 Interactive Audio Recorder and Transcriber",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------
# Sidebar - Settings
# -------------------------

st.sidebar.header("⚙️ Settings")

# Language selection for transcription
language = st.sidebar.selectbox(
    "Select Language for Transcription",
    [
        "English (US) - en-US",
        "Spanish (Spain) - es-ES",
        "French (France) - fr-FR",
        "German (Germany) - de-DE",
        "Italian (Italy) - it-IT",
        "Chinese (Simplified) - zh-CN",
        "Japanese - ja-JP",
        "Korean - ko-KR",
    ],
    index=0
)

# Extract language code
language_code = language.split(" - ")[-1]

# -------------------------
# Main App
# -------------------------

# Title and Description
st.title("🎤 Interactive Audio Recorder and Transcriber")
st.write(
    """
    Record your message using the audio recorder below. The app will display a waveform visualization 
    of your recording and transcribe the audio to text using Google's Speech Recognition API.
    """
)

# Audio Recorder
st.header("📻 Record Your Message")
audio_data = st.audio_input(
    "Click the button below to start recording:",
    type=["wav", "mp3", "m4a", "ogg"]
)

if audio_data:
    # Read audio bytes into memory
    audio_bytes = audio_data.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Display Waveform Visualization
    st.subheader("🔊 Waveform Visualization")
    try:
        options = WaveSurferOptions(
            wave_color="#2B88D9",
            progress_color="#b91d47",
            height=150,
            cursor_color="#FF5733",
            backend="WebAudio"
        )
        audix(audio_buffer, wavesurfer_options=options)
    except Exception as e:
        st.error(f"Error displaying waveform: {e}")

    # Provide Download Option
    st.download_button(
        label="💾 Download Audio",
        data=audio_bytes,
        file_name="recorded_audio.wav",
        mime="audio/wav"
    )

    # Transcribe Audio
    st.subheader("📝 Transcription")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_buffer) as source:
        audio = recognizer.record(source)
        with st.spinner("Transcribing your audio..."):
            try:
                transcription = recognizer.recognize_google(audio, language=language_code)
                st.success("Transcription successful!")
                st.write(transcription)

                # Copy to Clipboard Button
                st.download_button(
                    label="📋 Copy Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

            except sr.UnknownValueError:
                st.error("❌ Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"❌ Could not request results from Google Speech Recognition service; {e}")
