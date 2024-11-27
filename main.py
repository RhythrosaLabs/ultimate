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

# Live Preview Toggle
live_preview = st.checkbox("Enable Live Preview", value=False)

# Audio recording/upload section using the new st.audio_input
audio_input = st.audio_input("Record or upload your audio file")

if audio_input is not None:
    # Read audio data
    try:
        # st.audio_input returns a BytesIO object; read it with soundfile
        audio_input.seek(0)
        audio_data, samplerate = sf.read(audio_input)
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
        st.stop()

    # Ensure mono and float32 format for compatibility
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)

    st.success("Audio successfully uploaded or recorded!")

    # Playback original audio
    st.subheader("Playback Original Audio")
    st.audio(audio_input, format='audio/wav')

    # User controls for effects
    st.sidebar.title("Audio Effects")

    # Presets
    preset_choice = st.sidebar.selectbox("Choose an Effect Preset", ["None"] + list(PRESETS.keys()))
    if preset_choice != "None":
        preset_settings = PRESETS[preset_choice]
        # Initialize all possible effect variables to False or default
        apply_lowpass = False
        apply_highpass = False
        apply_reverb = False
        reverb_decay = 0.5
        apply_bitcrusher = False
        bit_depth = 8
        reverse_audio = False
        apply_speed_change = False
        speed_factor = 1.0
        apply_pitch_shift = False
        pitch_shift_steps = 0
        apply_amplify = False
        amplification_factor = 1.0
        apply_echo = False
        echo_delay = 500
        echo_decay = 0.5
        apply_chorus_effect = False
        apply_phaser_effect = False
        apply_overdrive_effect = False

        # Update settings based on preset
        for key, value in preset_settings.items():
            locals()[key] = value
    else:
        apply_lowpass = st.sidebar.checkbox("Apply Lowpass Filter")
        if apply_lowpass:
            cutoff_freq_low = st.sidebar.slider("Lowpass Cutoff Frequency (Hz)", 100, samplerate // 2, 1000)
        else:
            cutoff_freq_low = 1000

        apply_highpass = st.sidebar.checkbox("Apply Highpass Filter")
        if apply_highpass:
            cutoff_freq_high = st.sidebar.slider("Highpass Cutoff Frequency (Hz)", 20, samplerate // 2, 500)
        else:
            cutoff_freq_high = 500

        apply_reverb = st.sidebar.checkbox("Add Reverb")
        if apply_reverb:
            reverb_decay = st.sidebar.slider("Reverb Decay Factor", 0.1, 1.0, 0.5)
        else:
            reverb_decay = 0.5

        apply_bitcrusher = st.sidebar.checkbox("Apply Bitcrusher")
        if apply_bitcrusher:
            bit_depth = st.sidebar.slider("Bit Depth", 4, 16, 8)
        else:
            bit_depth = 8

        reverse_audio = st.sidebar.checkbox("Reverse Audio")

        apply_speed_change = st.sidebar.checkbox("Change Speed")
        if apply_speed_change:
            speed_factor = st.sidebar.slider("Speed Factor", 0.5, 2.0, 1.0)
        else:
            speed_factor = 1.0

        apply_pitch_shift = st.sidebar.checkbox("Pitch Shift")
        if apply_pitch_shift:
            pitch_shift_steps = st.sidebar.slider("Pitch Shift Steps", -12, 12, 0)
        else:
            pitch_shift_steps = 0

        apply_amplify = st.sidebar.checkbox("Amplify Volume")
        if apply_amplify:
            amplification_factor = st.sidebar.slider("Amplification Factor", 0.5, 3.0, 1.0)
        else:
            amplification_factor = 1.0

        apply_echo = st.sidebar.checkbox("Add Echo")
        if apply_echo:
            echo_delay = st.sidebar.slider("Echo Delay (ms)", 100, 2000, 500)
            echo_decay = st.sidebar.slider("Echo Decay Factor", 0.1, 1.0, 0.5)
        else:
            echo_delay = 500
            echo_decay = 0.5

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

        st.sidebar.success("Effects randomized!")

    # Apply effects
    processed_audio = audio_data

    if apply_lowpass:
        try:
            processed_audio = butter_lowpass_filter(processed_audio, cutoff_freq_low, samplerate)
            st.sidebar.success(f"Lowpass filter applied at {cutoff_freq_low} Hz")
        except Exception as e:
            st.sidebar.error(f"Error applying lowpass filter: {e}")

    if apply_highpass:
        try:
            processed_audio = butter_highpass_filter(processed_audio, cutoff_freq_high, samplerate)
            st.sidebar.success(f"Highpass filter applied at {cutoff_freq_high} Hz")
        except Exception as e:
            st.sidebar.error(f"Error applying highpass filter: {e}")

    if apply_reverb:
        try:
            processed_audio = add_reverb(processed_audio, reverb_decay)
            st.sidebar.success(f"Reverb added with decay factor {reverb_decay}")
        except Exception as e:
            st.sidebar.error(f"Error adding reverb: {e}")

    if apply_bitcrusher:
        try:
            processed_audio = bitcrusher(processed_audio, bit_depth)
            st.sidebar.success(f"Bitcrusher applied with bit depth {bit_depth}")
        except Exception as e:
            st.sidebar.error(f"Error applying bitcrusher: {e}")

    if reverse_audio:
        try:
            processed_audio = processed_audio[::-1]
            st.sidebar.success("Audio reversed")
        except Exception as e:
            st.sidebar.error(f"Error reversing audio: {e}")

    if apply_speed_change:
        try:
            processed_audio = librosa.effects.time_stretch(processed_audio, speed_factor)
            st.sidebar.success(f"Speed changed by a factor of {speed_factor}")
        except Exception as e:
            st.sidebar.error(f"Error changing speed: {e}")

    if apply_pitch_shift:
        try:
            processed_audio = librosa.effects.pitch_shift(processed_audio, samplerate, n_steps=pitch_shift_steps)
            st.sidebar.success(f"Pitch shifted by {pitch_shift_steps} semitones")
        except Exception as e:
            st.sidebar.error(f"Error applying pitch shift: {e}")

    if apply_amplify:
        try:
            processed_audio = processed_audio * amplification_factor
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            st.sidebar.success(f"Volume amplified by a factor of {amplification_factor}")
        except Exception as e:
            st.sidebar.error(f"Error amplifying volume: {e}")

    if apply_echo:
        try:
            delay_samples = int((echo_delay / 1000) * samplerate)
            echo = np.zeros_like(processed_audio)
            for i in range(delay_samples, len(processed_audio)):
                echo[i] = processed_audio[i] + echo_decay * processed_audio[i - delay_samples]
            processed_audio = echo
            st.sidebar.success(f"Echo added with delay {echo_delay} ms and decay factor {echo_decay}")
        except Exception as e:
            st.sidebar.error(f"Error adding echo: {e}")

    if apply_chorus_effect:
        try:
            processed_audio = apply_chorus(processed_audio, samplerate)
            st.sidebar.success("Chorus effect applied")
        except Exception as e:
            st.sidebar.error(f"Error applying chorus effect: {e}")

    if apply_phaser_effect:
        try:
            processed_audio = apply_phaser(processed_audio, samplerate)
            st.sidebar.success("Phaser effect applied")
        except Exception as e:
            st.sidebar.error(f"Error applying phaser effect: {e}")

    if apply_overdrive_effect:
        try:
            processed_audio = apply_overdrive(processed_audio)
            st.sidebar.success("Overdrive effect applied")
        except Exception as e:
            st.sidebar.error(f"Error applying overdrive effect: {e}")

    # Normalize the processed audio to prevent clipping
    try:
        audio_segment = AudioSegment(
            (processed_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=samplerate,
            sample_width=2,  # 16 bits
            channels=1
        )
        normalized_audio = normalize(audio_segment)
        processed_audio = np.array(normalized_audio.get_array_of_samples()).astype(np.float32) / 32767.0
    except Exception as e:
        st.sidebar.error(f"Error normalizing audio: {e}")

    # Convert processed audio to bytes for playback and download
    try:
        buffer = io.BytesIO()
        sf.write(buffer, processed_audio, samplerate, format='WAV')
        buffer.seek(0)
    except Exception as e:
        st.error(f"Error preparing audio for playback/download: {e}")
        st.stop()

    # Live Preview
    if live_preview:
        st.subheader("Live Preview - Processed Audio")
        st.audio(buffer, format='audio/wav')
        buffer.seek(0)  # Reset buffer position after playback

    # Download processed audio
    st.subheader("Download Processed Audio")
    st.download_button(
        label="Download WAV",
        data=buffer,
        file_name="processed_audio.wav",
        mime="audio/wav"
    )

    # Visualization (Optional)
    st.subheader("Waveform Comparison")
    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(audio_data, color='blue')
        ax[0].set_title("Original Audio")
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(processed_audio, color='orange')
        ax[1].set_title("Processed Audio")
        ax[1].set_xlabel("Samples")
        ax[1].set_ylabel("Amplitude")

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating waveform visualization: {e}")

else:
    st.info("Please upload or record an audio file to get started.")
