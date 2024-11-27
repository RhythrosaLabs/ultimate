import streamlit as st
import soundfile as sf
import numpy as np
import io
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft
import librosa

# Utility function for filtering audio
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# App configuration
st.set_page_config(
    page_title="Superpowered Audio Studio",
    page_icon="\U0001F3A7",
    layout="wide"
)

st.title("\U0001F3A7 Superpowered Audio Studio")
st.markdown("Record, enhance, and apply effects to your audio with ease!")

# Audio recording section
audio_input = st.audio_input("Record your audio")

if audio_input:
    audio_data, samplerate = sf.read(audio_input)
    st.success("Audio successfully uploaded!")

    # Playback original audio
    st.subheader("Playback Original Audio")
    st.audio(audio_input)

    # User control for effects
    st.sidebar.title("Audio Effects")
    apply_lowpass = st.sidebar.checkbox("Apply Lowpass Filter")
    cutoff_freq = st.sidebar.slider("Lowpass Cutoff Frequency (Hz)", 100, samplerate // 2, 1000)
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

    # Apply effects
    processed_audio = audio_data

    if apply_lowpass:
        processed_audio = butter_lowpass_filter(processed_audio, cutoff_freq, samplerate)
        st.sidebar.success(f"Lowpass filter applied at {cutoff_freq} Hz")

    if reverse_audio:
        processed_audio = processed_audio[::-1]
        st.sidebar.success("Audio reversed!")

    if apply_speed_change:
        indices = np.round(np.arange(0, len(processed_audio), speed_factor)).astype(int)
        indices = indices[indices < len(processed_audio)]
        processed_audio = processed_audio[indices]
        st.sidebar.success(f"Audio speed changed by a factor of {speed_factor}")

    if apply_pitch_shift:
        processed_audio = librosa.effects.pitch_shift(processed_audio.astype(np.float32), samplerate, n_steps=pitch_shift_steps)
        st.sidebar.success(f"Pitch shifted by {pitch_shift_steps} steps")

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
