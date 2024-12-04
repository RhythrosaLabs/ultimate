import streamlit as st
import wave
import numpy as np
from streamlit.components.v1 import html

# Set the title of the Streamlit app
st.title("🎙️ Audio Recorder and Playback")

# Add a stylish description with Apple-inspired design
html_description = """
<div style="background-color:#f5f5f7;padding:20px;border-radius:20px;box-shadow:0 4px 8px rgba(0, 0, 0, 0.1);">
    <h1 style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#1d1d1f;">Welcome to the Audio Recorder</h1>
    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;">Seamlessly record, analyze, and save your audio files with a sleek, modern interface.</p>
</div>
"""
html(html_description, height=180)

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
                <div style="background-color:#ffffff;padding:20px;border-radius:15px;box-shadow:0 4px 8px rgba(0, 0, 0, 0.1);">
                    <h2 style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#1d1d1f;">Audio Analysis</h2>
                    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;"><strong>Duration:</strong> {n_frames / frame_rate:.2f} seconds</p>
                    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;"><strong>Sample Rate:</strong> {frame_rate} Hz</p>
                    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;"><strong>Channels:</strong> {n_channels}</p>
                    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;"><strong>Sample Width:</strong> {sample_width} bytes</p>
                </div>
                """
                html(analysis_html, height=250)

                # Compute amplitude (for visualization purposes)
                audio_signal = np.frombuffer(frames, dtype=np.int16)
                st.line_chart(audio_signal[:min(1000, len(audio_signal))])

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")
