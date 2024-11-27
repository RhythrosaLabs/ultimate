import streamlit as st
import soundfile as sf
import numpy as np
import io
from scipy.signal import butter, lfilter
import librosa
from pydub.effects import normalize
from pydub.generators import Sine
from pydub import AudioSegment
import random

# Utility functions for filtering audio
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def add_reverb(audio_data, decay_factor=0.5):
    reverb = np.zeros_like(audio_data)
    for i in range(1, len(audio_data)):
        reverb[i] = audio_data[i] + decay_factor * reverb[i - 1]
    return reverb

def bitcrusher(audio_data, bit_depth=8):
    max_amplitude = np.max(np.abs(audio_data))
    step = max_amplitude / (2**bit_depth)
    crushed_audio = np.round(audio_data / step) * step
    return crushed_audio

def apply_chorus(audio_data, samplerate):
    delay_samples = int(0.02 * samplerate)  # 20ms delay
    chorus = np.zeros_like(audio_data)
    for i in range(delay_samples, len(audio_data)):
        chorus[i] = audio_data[i] + 0.5 * audio_data[i - delay_samples]
    return chorus

def apply_phaser(audio_data, samplerate):
    phase_shift = np.sin(np.linspace(0, np.pi * 2, len(audio_data)))
    return audio_data * (1 + 0.5 * phase_shift)

def apply_overdrive(audio_data):
    return np.clip(audio_data * 2, -1, 1)

# Preset effect chains
PRESETS = {
    "Smooth Vocals": {
        "apply_lowpass": True, "cutoff_freq_low": 500,
        "apply_reverb": True, "reverb_decay": 0.7,
    },
    "Lo-Fi Beat": {
        "apply_bitcrusher": True, "bit_depth": 6,
        "apply_chorus_effect": True,
    },
    "Psychedelic": {
        "apply_phaser_effect": True,
        "apply_echo": True, "echo_delay": 800, "echo_decay": 0.6,
    }
}

# App configuration
st.set_page_config(
    page_title="Superpowered Audio Studio",
    page_icon="\U0001F3A7",
    layout="wide"
)

st.title("\U0001F3A7 Superpowered Audio Studio")
st.markdown("Record, enhance, and apply effects to your audio with ease!")

# Audio recording section
audio_input = st.audio_input("Record or upload your audio file")

if audio_input:
    # Read audio data
    audio_data, samplerate = sf.read(audio_input)

    # Ensure mono and float32 format for compatibility
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)

    st.success("Audio successfully uploaded or recorded!")

    # Playback original audio
    st.subheader("Playback Original Audio")
    st.audio(audio_input)

    # User controls for effects
    st.sidebar.title("Audio Effects")

    # Presets
    preset_choice = st.sidebar.selectbox("Choose an Effect Preset", ["None"] + list(PRESETS.keys()))
    if preset_choice != "None":
        preset_settings = PRESETS[preset_choice]
        for key, value in preset_settings.items():
            locals()[key] = value
    else:
        apply_lowpass = st.sidebar.checkbox("Apply Lowpass Filter")
        cutoff_freq_low = st.sidebar.slider("Lowpass Cutoff Frequency (Hz)", 100, samplerate // 2, 1000)
        apply_highpass = st.sidebar.checkbox("Apply Highpass Filter")
        cutoff_freq_high = st.sidebar.slider("Highpass Cutoff Frequency (Hz)", 20, samplerate // 2, 500)
        apply_reverb = st.sidebar.checkbox("Add Reverb")
        reverb_decay = st.sidebar.slider("Reverb Decay Factor", 0.1, 1.0, 0.5)
        apply_bitcrusher = st.sidebar.checkbox("Apply Bitcrusher")
        bit_depth = st.sidebar.slider("Bit Depth", 4, 16, 8)
        reverse_audio = st.sidebar.checkbox("Reverse Audio")
        apply_speed_change = st.sidebar.checkbox("Change Speed")
        speed_factor = st.sidebar.slider("Speed Factor", 0.5, 2.0, 1.0)
        apply_pitch_shift = st.sidebar.checkbox("Pitch Shift")
        pitch_shift_steps = st.sidebar.slider("Pitch Shift Steps", -12, 12, 0)
        apply_amplify = st.sidebar.checkbox("Amplify Volume")
        amplification_factor = st.sidebar.slider("Amplification Factor", 0.5, 3.0, 1.0)
        apply_echo = st.sidebar.checkbox("Add Echo")
        echo_delay = st.sidebar.slider("Echo Delay (ms)", 100, 2000, 500)
        echo_decay = st.sidebar.slider("Echo Decay Factor", 0.1, 1.0, 0.5)
        apply_chorus_effect = st.sidebar.checkbox("Add Chorus")
        apply_phaser_effect = st.sidebar.checkbox("Add Phaser")
        apply_overdrive_effect = st.sidebar.checkbox("Add Overdrive")

    # Randomize effects
    st.sidebar.title("Randomize Effects")
    craziness = st.sidebar.slider("Craziness Level", 0.1, 1.0, 0.5)
    if st.sidebar.button("Randomize Effects"):
        apply_lowpass = random.random() < craziness
        cutoff_freq_low = random.randint(100, samplerate // 2) if apply_lowpass else 1000
        apply_highpass = random.random() < craziness
        cutoff_freq_high = random.randint(20, samplerate // 2) if apply_highpass else 500
        apply_reverb = random.random() < craziness
        reverb_decay = random.uniform(0.1, 1.0) if apply_reverb else 0.5
        apply_bitcrusher = random.random() < craziness
        bit_depth = random.randint(4, 16) if apply_bitcrusher else 8
        reverse_audio = random.random() < craziness
        apply_speed_change = random.random() < craziness
        speed_factor = random.uniform(0.5, 2.0) if apply_speed_change else 1.0
        apply_pitch_shift = random.random() < craziness
        pitch_shift_steps = random.randint(-12, 12) if apply_pitch_shift else 0
        apply_amplify = random.random() < craziness
        amplification_factor = random.uniform(0.5, 3.0) if apply_amplify else 1.0
        apply_echo = random.random() < craziness
        echo_delay = random.randint(100, 2000) if apply_echo else 500
        echo_decay = random.uniform(0.1, 1.0) if apply_echo else 0.5
        apply_chorus_effect = random.random() < craziness
        apply_phaser_effect = random.random() < craziness
        apply_overdrive_effect = random.random() < craziness

    # Apply effects
    processed_audio = audio_data

    if apply_lowpass:
        processed_audio = butter_lowpass_filter(processed_audio, cutoff_freq_low, samplerate)
        st.sidebar.success(f"Lowpass filter applied at {cutoff_freq_low} Hz")

    if apply_highpass:
        processed_audio = butter_highpass_filter(processed_audio, cutoff_freq_high, samplerate)
        st.sidebar.success(f"Highpass filter applied at {cutoff_freq_high} Hz")

    if apply_reverb:
        processed_audio = add_reverb(processed_audio, reverb_decay)
        st.sidebar.success(f"Reverb added with decay factor {reverb_decay}")

    if apply_bitcrusher:
        processed_audio = bitcrusher(processed_audio, bit_depth)
        st.sidebar.success(f"Bitcrusher applied with {bit_depth}-bit depth")

    if reverse_audio:
        processed
