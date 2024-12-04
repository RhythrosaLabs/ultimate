import streamlit as st
import wave
import numpy as np
from scipy.signal import butter, lfilter
from streamlit.components.v1 import html

# Helper function to apply audio effects
def apply_effect(effect_type, audio_signal, sample_rate):
    if effect_type == "Low Pass Filter":
        b, a = butter(5, 0.1, btype='low', fs=sample_rate)
        return lfilter(b, a, audio_signal)
    elif effect_type == "High Pass Filter":
        b, a = butter(5, 0.1, btype='high', fs=sample_rate)
        return lfilter(b, a, audio_signal)
    elif effect_type == "Amplify":
        return audio_signal * 2
    return audio_signal

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

# Input monitoring toggle
monitoring = st.checkbox("Enable Input Monitoring")

if monitoring and audio_data:
    st.audio(audio_data, format='audio/wav')

# If audio is recorded, process further
if audio_data:
    st.write("### Step 2: Play back your recording")
    st.audio(audio_data, format='audio/wav')

    # Save option
    st.write("### Step 3: Save, analyze, or add effects to your recording")
    save_option = st.checkbox("Save this recording")
    if save_option:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        st.success("✅ Audio recording saved as 'recorded_audio.wav'")

    # Analyze option
    analyze_option = st.checkbox("Analyze this recording")
    if analyze_option:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
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

    # Add effects option
    effect_option = st.selectbox("Choose an effect to apply:", ["None", "Low Pass Filter", "High Pass Filter", "Amplify"])
    if effect_option != "None":
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        with wave.open("temp_audio.wav", "rb") as wav_file:
            frames = wav_file.readframes(-1)
            frame_rate = wav_file.getframerate()
            audio_signal = np.frombuffer(frames, dtype=np.int16)
            processed_signal = apply_effect(effect_option, audio_signal, frame_rate)
            st.write(f"Effect Applied: {effect_option}")
            st.line_chart(processed_signal[:min(1000, len(processed_signal))])
