import streamlit as st
import wave
import numpy as np
from scipy.signal import butter, lfilter, hilbert
from scipy.fftpack import fft
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
    elif effect_type == "Echo":
        echo_signal = np.zeros(len(audio_signal) + sample_rate)
        echo_signal[:len(audio_signal)] = audio_signal
        echo_signal[sample_rate:] += audio_signal * 0.6
        return echo_signal[:len(audio_signal)]
    elif effect_type == "Reverb":
        reverb_signal = np.convolve(audio_signal, np.ones(500) / 500, mode='same')
        return reverb_signal
    elif effect_type == "Distortion":
        return np.clip(audio_signal, -10000, 10000)
    elif effect_type == "FFT Filter":
        transformed = fft(audio_signal)
        filtered = np.where(np.abs(transformed) > 500000, transformed, 0)
        return np.real(np.fft.ifft(filtered))
    elif effect_type == "Envelope Modulation":
        analytic_signal = hilbert(audio_signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope
    return audio_signal

# Set the title of the Streamlit app
st.title("🎙️ Audio Recorder and Playback")

# Add a stylish description with Apple-inspired design
html_description = """
<div style="background-color:#f5f5f7;padding:20px;border-radius:20px;box-shadow:0 4px 8px rgba(0, 0, 0, 0.1);">
    <h1 style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#1d1d1f;">Welcome to the Audio Recorder</h1>
    <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;">Record, analyze, and enhance your audio with a simple, intuitive interface.</p>
</div>
"""
html(html_description, height=180)

# Sidebar description
st.sidebar.title("🎛️ Audio Processing Tools")
st.sidebar.write("Enhance your audio with effects and customizations.")

# Step 1: Record Audio
st.write("### Step 1: Record your audio")
audio_data = st.audio_input("Please record your message:")

# Step 2: Playback and Monitoring
if audio_data:
    st.write("### Step 2: Play back your recording")
    st.audio(audio_data, format='audio/wav')

    monitoring = st.checkbox("Enable Input Monitoring", help="Listen to your input while recording")
    if monitoring:
        st.audio(audio_data, format='audio/wav')

    # Step 3: Save Audio
    st.write("### Step 3: Save or analyze your recording")
    save_option = st.button("Save Recording")
    if save_option:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        st.success("✅ Audio recording saved as 'recorded_audio.wav'")

    # Step 4: Analyze Audio
    if st.button("Analyze Recording"):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        try:
            with wave.open("temp_audio.wav", "rb") as wav_file:
                frames = wav_file.readframes(-1)
                frame_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                st.write("#### Audio Analysis")
                st.write(f"- **Duration:** {n_frames / frame_rate:.2f} seconds")
                st.write(f"- **Sample Rate:** {frame_rate} Hz")
                st.write(f"- **Channels:** {n_channels}")
                st.write(f"- **Sample Width:** {sample_width} bytes")

                # Visualization
                audio_signal = np.frombuffer(frames, dtype=np.int16)
                st.line_chart(audio_signal[:min(1000, len(audio_signal))])

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {e}")

    # Step 5: Apply Effects
    st.write("### Step 4: Apply effects to your recording")
    effect_option = st.sidebar.selectbox(
        "Choose an effect to apply:",
        ["None", "Low Pass Filter", "High Pass Filter", "Amplify", "Echo", "Reverb", "Distortion", "FFT Filter", "Envelope Modulation"],
        help="Select an audio effect to enhance your recording."
    )
    if effect_option != "None":
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        with wave.open("temp_audio.wav", "rb") as wav_file:
            frames = wav_file.readframes(-1)
            frame_rate = wav_file.getframerate()
            audio_signal = np.frombuffer(frames, dtype=np.int16)
            processed_signal = apply_effect(effect_option, audio_signal, frame_rate)

            st.write(f"### Effect Applied: {effect_option}")
            st.line_chart(processed_signal[:min(1000, len(processed_signal))])

            # Save processed audio
            if st.sidebar.button("Save Processed Audio"):
                with wave.open("processed_audio.wav", "wb") as processed_file:
                    processed_file.setnchannels(1)
                    processed_file.setsampwidth(2)  # Assuming 16-bit audio
                    processed_file.setframerate(frame_rate)
                    processed_file.writeframes(processed_signal.astype(np.int16).tobytes())
                st.success("✅ Processed audio saved as 'processed_audio.wav'")
