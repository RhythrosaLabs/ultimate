# Import necessary libraries
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import soundfile as sf

# Page configuration
st.set_page_config(
    page_title="Advanced Audio Recorder",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance UI/UX
custom_css = """
<style>
/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom font and background */
body {
    font-family: 'Helvetica Neue', sans-serif;
    background-color: #f0f2f6;
}

/* Header styling */
h1 {
    color: #333333;
    text-align: center;
    margin-bottom: 0px;
}

h2 {
    color: #333333;
    margin-top: 30px;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #ffffff;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
}

/* Tabs styling */
.css-1x8cf1d e1fqkh3o3 {
    background-color: #ffffff;
    border-bottom: 1px solid #e6e6e6;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>🎶 Advanced Audio Recorder with Effect Stacking</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; font-size: 18px; color: #666666;'>
    Record, play, and enhance your audio with multiple effects applied simultaneously.
    </p>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'recordings' not in st.session_state:
    st.session_state.recordings = []

if 'saved_session' not in st.session_state:
    st.session_state.saved_session = None

# Record audio
st.markdown("<h2>1️⃣ Record Audio</h2>", unsafe_allow_html=True)
audio_file = st.file_uploader("Upload your audio file (WAV format):", type=['wav'])

if audio_file:
    # Read audio data
    audio_bytes = audio_file.read()
    audio_data, sr_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Save recording in session state
    recording_name = f"Recording {len(st.session_state.recordings) + 1}"
    st.session_state.recordings.append({
        'audio_bytes': audio_bytes,
        'audio_data': audio_data,
        'sr_rate': sr_rate,
        'name': recording_name
    })
    st.success(f"{recording_name} saved!")

# Recording Management
st.markdown("<h2>2️⃣ Manage Recordings</h2>", unsafe_allow_html=True)
if st.session_state.recordings:
    recording_names = [rec['name'] for rec in st.session_state.recordings]
    selected_recording = st.selectbox("Select a recording to work with:", recording_names)
    rec_index = recording_names.index(selected_recording)
    audio_data = st.session_state.recordings[rec_index]['audio_data']
    sr_rate = st.session_state.recordings[rec_index]['sr_rate']
    audio_bytes = st.session_state.recordings[rec_index]['audio_bytes']
else:
    st.info("No recordings available. Please upload an audio file to use this feature.")
    audio_data = None
    sr_rate = None
    audio_bytes = None

# Sidebar for audio effects
st.sidebar.markdown("<h2>🎚️ Audio Effects</h2>", unsafe_allow_html=True)

# Effect selection
st.sidebar.markdown("<h3>Select Effects to Apply</h3>", unsafe_allow_html=True)
effects = {
    "Reverb": st.sidebar.checkbox("Reverb"),
    "Echo": st.sidebar.checkbox("Echo"),
    "Pitch Shift": st.sidebar.checkbox("Pitch Shift"),
    "Speed Change": st.sidebar.checkbox("Speed Change"),
    "Equalization": st.sidebar.checkbox("Equalization"),
    "Chorus": st.sidebar.checkbox("Chorus"),
    "Flanger": st.sidebar.checkbox("Flanger"),
    "Distortion": st.sidebar.checkbox("Distortion"),
    "Compression": st.sidebar.checkbox("Compression"),
    "Phaser": st.sidebar.checkbox("Phaser"),
    "Wah-Wah": st.sidebar.checkbox("Wah-Wah"),
}

# Effect parameters
st.sidebar.markdown("<h3>Adjust Effect Parameters</h3>", unsafe_allow_html=True)
params = {}

if effects["Reverb"]:
    params["reverb_amount"] = st.sidebar.slider("Reverb Amount", 0.0, 1.0, 0.5)

if effects["Echo"]:
    params["echo_delay"] = st.sidebar.slider("Echo Delay (ms)", 100, 1000, 500)
    params["echo_decay"] = st.sidebar.slider("Echo Decay", 0.0, 1.0, 0.5)

if effects["Pitch Shift"]:
    params["pitch_steps"] = st.sidebar.slider("Pitch Shift (semitones)", -12, 12, 0)

if effects["Speed Change"]:
    params["speed_rate"] = st.sidebar.slider("Speed Rate", 0.5, 2.0, 1.0)

if effects["Equalization"]:
    params["eq_low_gain"] = st.sidebar.slider("Low Frequency Gain (dB)", -12, 12, 0)
    params["eq_mid_gain"] = st.sidebar.slider("Mid Frequency Gain (dB)", -12, 12, 0)
    params["eq_high_gain"] = st.sidebar.slider("High Frequency Gain (dB)", -12, 12, 0)

if effects["Chorus"]:
    params["chorus_depth"] = st.sidebar.slider("Chorus Depth", 0.0, 1.0, 0.5)
    params["chorus_rate"] = st.sidebar.slider("Chorus Rate (Hz)", 0.1, 5.0, 1.5)

if effects["Flanger"]:
    params["flanger_depth"] = st.sidebar.slider("Flanger Depth", 0.0, 1.0, 0.5)
    params["flanger_rate"] = st.sidebar.slider("Flanger Rate (Hz)", 0.1, 5.0, 0.5)

if effects["Distortion"]:
    params["distortion_gain"] = st.sidebar.slider("Distortion Gain", 1.0, 20.0, 5.0)

if effects["Compression"]:
    params["compression_threshold"] = st.sidebar.slider("Compression Threshold (dB)", -60, 0, -20)
    params["compression_ratio"] = st.sidebar.slider("Compression Ratio", 1, 20, 4)

if effects["Phaser"]:
    params["phaser_rate"] = st.sidebar.slider("Phaser Rate (Hz)", 0.1, 5.0, 0.5)
    params["phaser_depth"] = st.sidebar.slider("Phaser Depth", 0.0, 1.0, 0.5)

if effects["Wah-Wah"]:
    params["wah_rate"] = st.sidebar.slider("Wah-Wah Rate (Hz)", 0.1, 5.0, 1.5)
    params["wah_depth"] = st.sidebar.slider("Wah-Wah Depth", 0.0, 1.0, 0.5)

# Placeholder for modified audio data
modified_audio_bytes = None

# Playback and visualization
st.markdown("<h2>3️⃣ Playback & Visualization</h2>", unsafe_allow_html=True)
original_tab, modified_tab = st.tabs(["Original", "Modified"])

with original_tab:
    st.markdown("<h3>🔊 Original Audio</h3>", unsafe_allow_html=True)
    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        fig_original, ax_original = plt.subplots()
        librosa.display.waveshow(audio_data, sr=sr_rate, ax=ax_original)
        ax_original.set_xlabel("Time (s)")
        ax_original.set_ylabel("Amplitude")
        st.pyplot(fig_original)
    else:
        st.info("No audio data available. Please upload an audio file.")

with modified_tab:
    st.markdown("<h3>🎧 Modified Audio</h3>", unsafe_allow_html=True)
    if audio_bytes:
        # Define effect functions
        def apply_reverb(data, amount):
            ir = np.zeros(int(sr_rate * 0.3))
            ir[0] = 1.0
            ir[int(len(ir) * 0.5):] = amount
            return np.convolve(data, ir, mode='full')[:len(data)]

        def apply_echo(data, delay_ms, decay):
            delay_samples = int(sr_rate * (delay_ms / 1000.0))
            echo_data = np.zeros(len(data) + delay_samples)
            echo_data[:len(data)] = data
            echo_data[delay_samples:] += data * decay
            return echo_data[:len(data)]

        def apply_pitch_shift(data, n_steps):
            return librosa.effects.pitch_shift(data, sr_rate, n_steps=n_steps)

        def apply_speed_change(data, rate):
            return librosa.effects.time_stretch(data, rate)

        def apply_equalization(data, low_gain, mid_gain, high_gain):
            S = librosa.stft(data)
            frequencies = librosa.fft_frequencies(sr=sr_rate)
            low_freqs = frequencies < 500
            mid_freqs = (frequencies >= 500) & (frequencies <= 2000)
            high_freqs = frequencies > 2000
            S[low_freqs, :] *= 10 ** (low_gain / 20)
            S[mid_freqs, :] *= 10 ** (mid_gain / 20)
            S[high_freqs, :] *= 10 ** (high_gain / 20)
            return librosa.istft(S)

        def apply_chorus(data, rate, depth):
            t = np.arange(len(data)) / sr_rate
            mod = depth * np.sin(2 * np.pi * rate * t)
            chorus_data = np.interp(t + mod, t, data, left=0, right=0)
            return data + chorus_data

        def apply_flanger(data, rate, depth):
            t = np.arange(len(data)) / sr_rate
            delay = depth * np.sin(2 * np.pi * rate * t)
            delay_samples = (delay * sr_rate).astype(int)
            flanged = np.copy(data)
            for i in range(len(data)):
                if i - delay_samples[i] >= 0:
                    flanged[i] += data[i - delay_samples[i]]
            return flanged / 2

        def apply_distortion(data, gain):
            return np.tanh(gain * data)

        def apply_compression(data, threshold_db, ratio):
            threshold = 10 ** (threshold_db / 20)
            compressed = np.copy(data)
            over_threshold = np.abs(data) > threshold
            compressed[over_threshold] = np.sign(data[over_threshold]) * (
                threshold + (np.abs(data[over_threshold]) - threshold) / ratio)
            return compressed

        def apply_phaser(data, rate, depth):
            t = np.arange(len(data)) / sr_rate
            phase = depth * np.sin(2 * np.pi * rate * t)
            phaser_data = np.copy(data)
            for i in range(1, len(data)):
                phaser_data[i] += phase[i] * data[i - 1]
            return phaser_data

        def apply_wahwah(data, rate, depth):
            t = np.arange(len(data)) / sr_rate
            wah = depth * np.sin(2 * np.pi * rate * t)
            wah_data = data * (1 + wah)
            return wah_data

        # Apply selected effects
        def apply_effects(data):
            modified = data.copy()
            if effects["Reverb"]:
                modified = apply_reverb(modified, params["reverb_amount"])
            if effects["Echo"]:
                modified = apply_echo(modified, params["echo_delay"], params["echo_decay"])
            if effects["Pitch Shift"]:
                modified = apply_pitch_shift(modified, params["pitch_steps"])
            if effects["Speed Change"]:
                modified = apply_speed_change(modified, params["speed_rate"])
            if effects["Equalization"]:
                modified = apply_equalization(modified, params["eq_low_gain"],
                                              params["eq_mid_gain"], params["eq_high_gain"])
            if effects["Chorus"]:
                modified = apply_chorus(modified, params["chorus_rate"], params["chorus_depth"])
            if effects["Flanger"]:
                modified = apply_flanger(modified, params["flanger_rate"], params["flanger_depth"])
            if effects["Distortion"]:
                modified = apply_distortion(modified, params["distortion_gain"])
            if effects["Compression"]:
                modified = apply_compression(modified, params["compression_threshold"],
                                             params["compression_ratio"])
            if effects["Phaser"]:
                modified = apply_phaser(modified, params["phaser_rate"], params["phaser_depth"])
            if effects["Wah-Wah"]:
                modified = apply_wahwah(modified, params["wah_rate"], params["wah_depth"])
            # Normalize to prevent clipping
            max_abs = np.max(np.abs(modified))
            if max_abs > 1.0:
                modified = modified / max_abs
            return modified

        with st.spinner("Applying effects..."):
            modified_data = apply_effects(audio_data)

        # Save modified audio to buffer
        modified_audio = io.BytesIO()
        sf.write(modified_audio, modified_data, sr_rate, format='wav')
        modified_audio_bytes = modified_audio.getvalue()

        st.audio(modified_audio_bytes, format='audio/wav')
        fig_modified, ax_modified = plt.subplots()
        librosa.display.waveshow(modified_data, sr=sr_rate, ax=ax_modified)
        ax_modified.set_xlabel("Time (s)")
        ax_modified.set_ylabel("Amplitude")
        st.pyplot(fig_modified)
    else:
        st.info("No audio data available. Please upload an audio file.")

# Session saving/loading
st.markdown("<h2>4️⃣ Session Management</h2>", unsafe_allow_html=True)
session_name = st.text_input("Session Name", value="My Session")
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save Session"):
        if st.session_state.recordings:
            session_data = {
                'recordings': st.session_state.recordings,
                'effects': effects,
                'params': params,
                'selected_recording': selected_recording if st.session_state.recordings else None
            }
            st.session_state.saved_session = session_data
            st.success("Session saved successfully!")
        else:
            st.error("No recordings to save.")
with col2:
    if st.button("📂 Load Session"):
        if st.session_state.saved_session:
            session_data = st.session_state.saved_session
            st.session_state.recordings = session_data['recordings']
            effects = session_data['effects']
            params = session_data['params']
            selected_recording = session_data['selected_recording']
            st.success("Session loaded successfully!")
        else:
            st.error("No saved session found.")

# Download option
st.markdown("<h2>5️⃣ Download</h2>", unsafe_allow_html=True)
if audio_bytes and modified_audio_bytes:
    st.download_button(
        label="💾 Download Modified Audio",
        data=modified_audio_bytes,
        file_name=f"{selected_recording}_modified.wav",
        mime="audio/wav"
    )
else:
    st.info("No audio data available to download.")
