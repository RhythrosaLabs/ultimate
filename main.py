import streamlit as st
import wave
import numpy as np
from streamlit.components.v1 import html

# Set the title of the Streamlit app
st.title("🎙️ Audio Recorder and Playback")

# Add a stylish description with HTML
html_description = """
<div style="background-color:#f0f8ff;padding:10px;border-radius:10px;">
    <h3 style="color:#4a90e2;">Welcome to the Audio Recorder App!</h3>
    <p style="color:#333;">Record your audio, play it back, analyze its properties, and save it for later use.</p>
</div>
"""
html(html_description, height=120)

# Display the audio input widget
st.write("### Step 1: Record your audio")
audio_data = st.audio_input("Please record your message:")

# If audio is recorded, play it back
if audio_data:
    st.write("### Step 2: Play back your recording")
    st.audio(audio_data, format='audio/wav')

    # Provide options for saving the audio and analyzing it
    st.write("### Step 3: Save or analyze your recording")

    save_option = st.checkbox("Save this recording")

    if save_option:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        st.success("✅ Audio recording saved as 'recorded_audio.wav'")

    analyze_option = st.checkbox("Analyze this recording")

    if analyze_option:
        # Save audio data temporarily for analysis
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())

        # Perform audio analysis
        try:
            with wave.open("temp_audio.wav", "rb") as wav_file:
                frames = wav_file.readframes(-1)
                frame_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                analysis_html = f"""
                <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;">
                    <h4 style="color:#388e3c;">Audio Analysis</h4>
                    <p><strong>Duration:</strong> {n_frames / frame_rate:.2f} seconds</p>
                    <p><strong>Sample Rate:</strong> {frame_rate} Hz</p>
                    <p><strong>Channels:</strong> {n_channels}</p>
                    <p><strong>Sample Width:</strong> {sample_width} bytes</p>
                </div>
                """
                html(analysis_html, height=200)

                # Compute amplitude (for visualization purposes)
                audio_signal = np.frombuffer(frames, dtype=np.int16)
                st.line_chart(audio_signal[:min(1000, len(audio_signal))])

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")
