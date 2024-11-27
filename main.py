import streamlit as st
import soundfile as sf
import numpy as np
import io
from scipy.signal import butter, lfilter
import librosa
from pydub.effects import normalize
from pydub.generators import Sine
from pydub import AudioSegment

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
        processed_audio = processed_audio[::-1]
        st.sidebar.success("Audio reversed!")

    if apply_speed_change:
        indices = np.round(np.arange(0, len(processed_audio), speed_factor)).astype(int)
        indices = indices[indices < len(processed_audio)]
        processed_audio = processed_audio[indices]
        st.sidebar.success(f"Audio speed changed by a factor of {speed_factor}")

    if apply_pitch_shift:
        try:
            processed_audio = librosa.effects.pitch_shift(processed_audio, samplerate, n_steps=pitch_shift_steps)
            st.sidebar.success(f"Pitch shifted by {pitch_shift_steps} steps")
        except Exception as e:
            st.sidebar.error(f"Error applying pitch shift: {e}")

    if apply_amplify:
        processed_audio = processed_audio * amplification_factor
        st.sidebar.success(f"Volume amplified by a factor of {amplification_factor}")

    if apply_echo:
        delay_samples = int((echo_delay / 1000.0) * samplerate)
        echo = np.zeros_like(processed_audio)
        for i in range(delay_samples, len(processed_audio)):
            echo[i] = processed_audio[i - delay_samples] * echo_decay
        processed_audio = processed_audio + echo
        st.sidebar.success(f"Echo added with {echo_delay} ms delay and {echo_decay} decay factor")

    if apply_chorus_effect:
        processed_audio = apply_chorus(processed_audio, samplerate)
        st.sidebar.success("Chorus effect applied!")

    if apply_phaser_effect:
        processed_audio = apply_phaser(processed_audio, samplerate)
        st.sidebar.success("Phaser effect applied!")

    if apply_overdrive_effect:
        processed_audio = apply_overdrive(processed_audio)
        st.sidebar.success("Overdrive effect applied!")

    # Save processed audio
    output_audio = io.BytesIO()
    sf.write(output_audio, processed_audio, samplerate, format='WAV')
    output_audio.seek(0)

    # Playback processed audio
    st.subheader("Playback Processed Audio")
    st.audio(output_audio)

    # Allow user to download the processed audio
    st.download_button("Download Processed Audio", output_audio, file_name="processed_audio.wav")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit \U0001F680 | Experiment with audio like never before!")
