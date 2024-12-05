import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, hilbert
import scipy.signal as signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import io
import librosa
import librosa.display
import random
import soundfile as sf
import warnings
import wave  # Added import for wave module
warnings.filterwarnings('ignore')

import os
import pickle
import tempfile
import zipfile

from streamlit.components.v1 import html

import plotly.express as px
import plotly.graph_objects as go

# ------------------------ Helper Functions ------------------------

# Audio Effects for Recorder
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

# Noise Generation Functions
def generate_white_noise(duration, sample_rate):
    return np.random.normal(0, 1, int(duration * sample_rate))

def generate_pink_noise(duration, sample_rate):
    samples = int(duration * sample_rate)
    n_rows = 16
    n_columns = int(np.ceil(samples / n_rows))
    array = np.random.randn(n_rows, n_columns)
    cumulative = np.cumsum(array, axis=0)
    pink_noise = cumulative[-1, :samples]
    return pink_noise

def generate_brown_noise(duration, sample_rate):
    samples = int(duration * sample_rate)
    brown_noise = np.cumsum(np.random.randn(samples))
    brown_noise /= np.max(np.abs(brown_noise) + 1e-7)
    return brown_noise

def generate_blue_noise(duration, sample_rate):
    samples = int(duration * sample_rate)
    white = np.random.normal(0, 1, samples)
    blue_noise = np.diff(white, prepend=0)
    return blue_noise

def generate_violet_noise(duration, sample_rate):
    samples = int(duration * sample_rate)
    white = np.random.normal(0, 1, samples)
    violet_noise = np.diff(np.diff(white, prepend=0), prepend=0)
    return violet_noise

def generate_grey_noise(duration, sample_rate):
    samples = int(duration * sample_rate)
    white = np.random.normal(0, 1, samples)
    freqs = np.fft.rfftfreq(samples, 1/sample_rate)
    a_weighting = (12200**2 * freqs**4) / ((freqs**2 + 20.6**2) * 
                np.sqrt((freqs**2 + 107.7**2) * (freqs**2 + 737.9**2)) * 
                (freqs**2 + 12200**2))
    a_weighting /= np.max(a_weighting + 1e-7)
    white_fft = np.fft.rfft(white)
    grey_fft = white_fft * a_weighting
    grey_noise = np.fft.irfft(grey_fft)
    return grey_noise

def generate_waveform(waveform_type, frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform_type == "Sine Wave":
        return np.sin(2 * np.pi * frequency * t)
    elif waveform_type == "Square Wave":
        return signal.square(2 * np.pi * frequency * t)
    elif waveform_type == "Sawtooth Wave":
        return signal.sawtooth(2 * np.pi * frequency * t)
    elif waveform_type == "Triangle Wave":
        return signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    else:
        return np.zeros_like(t)

def generate_synth_tone(note, duration, sample_rate, waveform='Sine', envelope=None):
    frequency = note_to_freq(note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform == 'Sine':
        tone = np.sin(2 * np.pi * frequency * t)
    elif waveform == 'Square':
        tone = signal.square(2 * np.pi * frequency * t)
    elif waveform == 'Sawtooth':
        tone = signal.sawtooth(2 * np.pi * frequency * t)
    elif waveform == 'Triangle':
        tone = signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    else:
        tone = np.zeros_like(t)
    if envelope:
        tone *= envelope
    return tone

def note_to_freq(note):
    A4_freq = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    if len(note) == 2:
        key = note[0]
        octave = int(note[1])
    elif len(note) == 3:
        key = note[0:2]
        octave = int(note[2])
    else:
        return A4_freq
    if key not in note_names:
        return A4_freq
    key_number = note_names.index(key)
    n = key_number - 9 + (octave - 4) * 12
    freq = A4_freq * (2 ** (n / 12))
    return freq

def generate_envelope(total_samples, sample_rate, attack, decay, sustain, release):
    envelope = np.zeros(total_samples)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    sustain_samples = max(sustain_samples, 0)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
    if sustain_samples > 0:
        envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
    return envelope

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_fade(data, sample_rate, fade_in, fade_out):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    total_samples = data.shape[0]
    fade_in_samples = int(fade_in * sample_rate)
    fade_out_samples = int(fade_out * sample_rate)
    fade_in_samples = min(fade_in_samples, total_samples)
    fade_out_samples = min(fade_out_samples, total_samples)
    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples).reshape(-1, 1)
        data[:fade_in_samples] *= fade_in_curve
    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples).reshape(-1, 1)
        data[-fade_out_samples:] *= fade_out_curve
    if data.shape[1] == 1:
        data = data.flatten()
    return data

def apply_amplitude_modulation(data, sample_rate, mod_freq=5.0):
    t = np.linspace(0, len(data)/sample_rate, num=len(data))
    modulator = np.sin(2 * np.pi * mod_freq * t)
    return data * modulator

def apply_frequency_modulation(data, sample_rate, mod_freq=5.0, mod_index=2.0):
    t = np.linspace(0, len(data)/sample_rate, num=len(data))
    carrier = np.sin(2 * np.pi * 440 * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
    return data * carrier

def apply_reverb(data, sample_rate, decay=0.5):
    reverb_data = np.copy(data)
    delay_samples = int(0.02 * sample_rate)
    for i in range(delay_samples, len(data)):
        reverb_data[i] += decay * reverb_data[i - delay_samples]
    return reverb_data

def apply_delay(data, sample_rate, delay_time=0.5, feedback=0.5):
    delay_samples = int(delay_time * sample_rate)
    delayed_data = np.zeros(len(data) + delay_samples)
    delayed_data[:len(data)] = data
    for i in range(len(data)):
        delayed_data[i + delay_samples] += data[i] * feedback
    return delayed_data[:len(data)]

def apply_distortion(data, gain=20, threshold=0.5):
    data = data * gain
    data = np.clip(data, -threshold, threshold)
    return data

def adjust_bit_depth(data, bit_depth):
    max_val = 2 ** (bit_depth - 1) - 1
    data = data * max_val
    data = np.round(data)
    data = data / max_val
    return data

def pan_stereo(data, panning):
    left = data * (1 - panning)
    right = data * panning
    stereo_data = np.vstack((left, right)).T
    return stereo_data

def generate_algorithmic_composition(duration, sample_rate, composition_type):
    if composition_type == "Random Melody":
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        melody = np.zeros_like(t)
        num_notes = int(duration * 2)
        note_durations = np.random.uniform(0.1, 0.5, size=num_notes)
        note_frequencies = np.random.uniform(200, 800, size=num_notes)
        current_sample = 0
        for i in range(num_notes):
            note_duration_samples = int(note_durations[i] * sample_rate)
            end_sample = current_sample + note_duration_samples
            if end_sample > len(t):
                end_sample = len(t)
            melody[current_sample:end_sample] = np.sin(2 * np.pi * note_frequencies[i] * t[current_sample:end_sample])
            current_sample = end_sample
            if current_sample >= len(t):
                break
        return melody
    elif composition_type == "Ambient Soundscape":
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        soundscape = np.sin(2 * np.pi * 220 * t) * np.random.uniform(0.5, 1.0)
        return soundscape
    elif composition_type == "Rhythmic Pattern":
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        bpm = 120
        beat_duration = 60 / bpm
        num_beats = int(duration / beat_duration)
        rhythm = np.zeros_like(t)
        for i in range(num_beats):
            start_sample = int(i * beat_duration * sample_rate)
            end_sample = start_sample + int(0.1 * sample_rate)
            if end_sample > len(t):
                end_sample = len(t)
            rhythm[start_sample:end_sample] = 1.0
        return rhythm * np.sin(2 * np.pi * 440 * t)
    else:
        return np.zeros(int(duration * sample_rate))

def pitch_shift_audio(data, sample_rate, n_steps):
    return librosa.effects.pitch_shift(data, sample_rate, n_steps=n_steps)

def apply_tremolo(data, sample_rate, rate=5.0, depth=0.8):
    t = np.arange(len(data)) / sample_rate
    tremolo = (1 + depth * np.sin(2 * np.pi * rate * t)) / 2
    return data * tremolo

def apply_bitcrusher(data, bit_depth):
    max_val = 2 ** (bit_depth - 1) - 1
    data = data * max_val
    data = np.round(data)
    data = data / max_val
    return data

def apply_sample_reduction(data, reduction_factor):
    reduced_data = data[::reduction_factor]
    upsampled_data = np.repeat(reduced_data, reduction_factor)
    upsampled_data = upsampled_data[:len(data)]
    return upsampled_data

def apply_reverse(data):
    return data[::-1]

def apply_stutter(data, sample_rate, interval=0.1):
    stutter_samples = int(interval * sample_rate)
    num_repeats = 3
    stuttered_data = []
    for i in range(0, len(data), stutter_samples):
        chunk = data[i:i+stutter_samples]
        for _ in range(num_repeats):
            stuttered_data.append(chunk)
    stuttered_data = np.concatenate(stuttered_data)
    return stuttered_data[:len(data)]

def apply_glitch(data, sample_rate):
    glitch_length = int(0.05 * sample_rate)
    glitch_data = np.copy(data)
    for i in range(0, len(data), glitch_length * 4):
        glitch_data[i:i+glitch_length] = 0
    return glitch_data

def apply_arpeggiation(data, sample_rate, pattern='Ascending'):
    num_notes = 4
    note_length = int(len(data) / num_notes)
    arpeggiated_data = np.zeros_like(data)
    indices = list(range(num_notes))
    if pattern == 'Ascending':
        pass
    elif pattern == 'Descending':
        indices = indices[::-1]
    elif pattern == 'Random':
        random.shuffle(indices)
    for idx, i in enumerate(indices):
        start = i * note_length
        end = start + note_length
        arpeggiated_data[start:end] = data[start:end]
    return arpeggiated_data

def apply_sequencer(data, sample_rate, pattern='Random'):
    sequence_length = int(sample_rate * 0.5)
    num_sequences = len(data) // sequence_length
    sequences = [data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]
    if pattern == 'Ascending':
        pass
    elif pattern == 'Descending':
        sequences = sequences[::-1]
    elif pattern == 'Random':
        random.shuffle(sequences)
    sequenced_data = np.concatenate(sequences)
    return sequenced_data

def apply_chorus(data, sample_rate, rate=1.5, depth=0.5):
    delay_samples = int((1 / rate) * sample_rate)
    modulated_delay = depth * np.sin(2 * np.pi * rate * np.arange(len(data)) / sample_rate)
    delayed_data = np.zeros_like(data)
    for i in range(len(data)):
        delay = int(delay_samples * (1 + modulated_delay[i]))
        if i - delay >= 0:
            delayed_data[i] = data[i - delay]
    return data + delayed_data

def apply_flanger(data, sample_rate, rate=0.5, depth=0.7):
    delay_samples = int(0.001 * sample_rate)
    modulated_delay = depth * delay_samples * np.sin(2 * np.pi * rate * np.arange(len(data)) / sample_rate)
    flanged_data = np.zeros_like(data)
    for i in range(len(data)):
        delay = int(modulated_delay[i])
        if i - delay >= 0:
            flanged_data[i] = data[i] + data[i - delay]
        else:
            flanged_data[i] = data[i]
    return flanged_data

def apply_phaser(data, sample_rate, rate=0.5, depth=0.7):
    phaser_data = np.copy(data)
    phase = depth * np.sin(2 * np.pi * rate * np.arange(len(data)) / sample_rate)
    phaser_data = np.sin(2 * np.pi * phaser_data + phase)
    return phaser_data

def apply_compression(data, threshold=0.5, ratio=2.0):
    compressed_data = np.copy(data)
    over_threshold = np.abs(compressed_data) > threshold
    compressed_data[over_threshold] = threshold + (compressed_data[over_threshold] - threshold) / ratio
    compressed_data[compressed_data < -threshold] = -threshold + (compressed_data[compressed_data < -threshold] + threshold) / ratio
    return compressed_data

def apply_eq(data, sample_rate, bands):
    eq_data = np.copy(data)
    # Low frequencies (20 Hz - 250 Hz)
    b, a = butter(2, [20 / (0.5 * sample_rate), 250 / (0.5 * sample_rate)], btype='band')
    low = lfilter(b, a, data) * (10 ** (bands.get('low', 0.0) / 20))
    # Mid frequencies (250 Hz - 4 kHz)
    b, a = butter(2, [250 / (0.5 * sample_rate), 4000 / (0.5 * sample_rate)], btype='band')
    mid = lfilter(b, a, data) * (10 ** (bands.get('mid', 0.0) / 20))
    # High frequencies (4 kHz - 20 kHz)
    b, a = butter(2, [4000 / (0.5 * sample_rate), 20000 / (0.5 * sample_rate)], btype='band')
    high = lfilter(b, a, data) * (10 ** (bands.get('high', 0.0) / 20))
    eq_data = low + mid + high
    return eq_data

def apply_pitch_shift(data, sample_rate, semitones):
    return librosa.effects.pitch_shift(data, sample_rate, n_steps=semitones)

def apply_highpass_filter(data, sample_rate, cutoff):
    b, a = butter(2, cutoff / (0.5 * sample_rate), btype='high')
    return lfilter(b, a, data)

def apply_lowpass_filter(data, sample_rate, cutoff):
    b, a = butter(2, cutoff / (0.5 * sample_rate), btype='low')
    return lfilter(b, a, data)

def apply_vibrato(data, sample_rate, rate=5.0, depth=0.5):
    t = np.arange(len(data))
    vibrato = np.sin(2 * np.pi * rate * t / sample_rate)
    indices = t + depth * vibrato * sample_rate
    indices = np.clip(indices, 0, len(data) - 1).astype(int)
    return data[indices]

def apply_autopan(data, sample_rate, rate=1.0):
    t = np.arange(len(data)) / sample_rate
    pan = 0.5 * (1 + np.sin(2 * np.pi * rate * t))
    left = data * pan
    right = data * (1 - pan)
    stereo_data = np.vstack((left, right)).T
    return stereo_data

# Preset Management
def save_preset(params, name):
    if not os.path.exists('presets'):
        os.makedirs('presets')
    with open(f'presets/{name}.pkl', 'wb') as f:
        pickle.dump(params, f)

def load_preset(name):
    with open(f'presets/{name}.pkl', 'rb') as f:
        params = pickle.load(f)
    return params

def list_presets():
    if os.path.exists('presets'):
        return [f.replace('.pkl', '') for f in os.listdir('presets') if f.endswith('.pkl')]
    else:
        return []

# Additional function to generate synth tone from frequency
def generate_synth_tone_from_freq(frequency, duration, sample_rate, waveform='Sine', envelope=None):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if waveform == 'Sine':
        tone = np.sin(2 * np.pi * frequency * t)
    elif waveform == 'Square':
        tone = signal.square(2 * np.pi * frequency * t)
    elif waveform == 'Sawtooth':
        tone = signal.sawtooth(2 * np.pi * frequency * t)
    elif waveform == 'Triangle':
        tone = signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    else:
        tone = np.zeros_like(t)
    if envelope is not None:
        tone *= envelope
    return tone

# ------------------------ Main Application ------------------------

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Industrial Noise and Audio Recorder",
        page_icon="🎹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
        /* Main layout */
        .main {
            background-color: #1a1a1a;
            color: white;
        }
        /* Buttons */
        .stButton>button {
            width: 100%;
            padding: 10px;
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
        .stDownloadButton>button {
            width: 100%;
            padding: 10px;
            background-color: #00cc66;
            color: white;
            border-radius: 10px;
            border: none;
        }
        .stDownloadButton>button:hover {
            background-color: #00994d;
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #FF4B4B;
        }
        /* Text input and select boxes */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>div>input {
            background-color: #3d3d3d;
            color: white;
        }
        /* Sliders */
        .stSlider>div>div>div {
            color: white;
        }
        /* File uploader */
        .stFileUploader>div>div {
            background-color: #3d3d3d;
            color: white;
        }
        /* Checkbox */
        .stCheckbox>div>div>div {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # Application Title and Description
    st.title("🎹 Industrial Noise Generator and 🎙️ Audio Recorder")
    st.markdown("""
    Welcome to the **Industrial Noise Generator and Audio Recorder** app! This application offers two main functionalities:

    - **Industrial Noise Generator Pro Max with Variations:** Generate multiple industrial noise samples with advanced customization options.
    
    - **Audio Recorder and Playback:** Upload, apply effects, analyze, and save your audio recordings.

    Use the tabs below to navigate between the two features.
    """)

    # Creating Tabs
    tabs = st.tabs(["Industrial Noise Generator", "Audio Recorder and Playback"])

    # -------------------- Tab 1: Industrial Noise Generator --------------------
    with tabs[0]:
        st.header("🎹 Industrial Noise Generator Pro Max with Variations")
        st.markdown("""
        This tool allows you to generate industrial noise samples with a variety of customization options:

        - **Signal Flow Customization:** Reroute effects and components in any order.
        - **BPM Customization:** Set beats per minute for rhythm-based noise.
        - **Note and Scale Selection:** Choose notes and scales for the synthesizer.
        - **Built-in Synthesizer:** Create unique sounds with various waveforms.
        - **Advanced Effects:** Apply reverb, delay, distortion, and more in any order.
        - **Variations:** Automatically generate variations for each sample.
        - **Interactive Waveform Editor:** Zoom, pan, and interact with your waveform.
        - **Session Management:** Store and download all your generated sounds.

        **Adjust the parameters below and click "Generate Noise" to create your samples!**
        """)

        # Sample Generation Controls
        st.subheader("🔢 Sample Generation")
        num_samples = st.number_input(
            "Number of Samples to Generate",
            min_value=1,
            max_value=100,
            value=1,
            help="Select how many unique samples you want to generate."
        )

        # Presets Selection
        st.subheader("🎚️ Presets")
        preset_options = ["Default", "Heavy Machinery", "Factory Floor", "Electric Hum", "Custom"]
        preset = st.selectbox(
            "Choose a Preset",
            preset_options,
            help="Select a preset to quickly load predefined settings."
        )

        # Function to set preset parameters
        def set_preset(preset):
            params = {}
            if preset == "Default":
                params = {
                    'duration_type': 'Seconds',
                    'duration': 5,
                    'beats': 8,
                    'bpm': 120,
                    'noise_types': ["White Noise"],
                    'waveform_types': [],
                    'lowcut': 100,
                    'highcut': 5000,
                    'order': 6,
                    'amplitude': 1.0,
                    'sample_rate': 48000,
                    'channels': "Mono",
                    'fade_in': 0.0,
                    'fade_out': 0.0,
                    'panning': 0.5,
                    'bit_depth': 16,
                    'modulation': None,
                    'uploaded_file': None,
                    'reverse_audio': False,
                    'bitcrusher': False,
                    'bitcrusher_depth': 8,
                    'sample_reduction': False,
                    'sample_reduction_factor': 1,
                    'rhythmic_effects': [],
                    'arpeggiation': False,
                    'sequencer': False,
                    'sequence_pattern': 'Random',
                    'effects_chain': [],
                    'effect_params': {},
                    'voice_changer': False,
                    'pitch_shift_semitones': 0,
                    'algorithmic_composition': False,
                    'composition_type': None,
                    'synth_enabled': False,
                    'synth_notes': ['C4'],
                    'synth_scale': 'Major',
                    'synth_waveform': 'Sine',
                    'synth_attack': 0.01,
                    'synth_decay': 0.1,
                    'synth_sustain': 0.7,
                    'synth_release': 0.2
                }
            elif preset == "Heavy Machinery":
                params = {
                    'duration_type': 'Seconds',
                    'duration': 10,
                    'beats': 16,
                    'bpm': 90,
                    'noise_types': ["Brown Noise"],
                    'waveform_types': ["Square Wave"],
                    'lowcut': 50,
                    'highcut': 3000,
                    'order': 8,
                    'amplitude': 0.9,
                    'sample_rate': 44100,
                    'channels': "Stereo",
                    'fade_in': 0.5,
                    'fade_out': 0.5,
                    'panning': 0.6,
                    'bit_depth': 24,
                    'modulation': "Amplitude Modulation",
                    'uploaded_file': None,
                    'reverse_audio': False,
                    'bitcrusher': True,
                    'bitcrusher_depth': 6,
                    'sample_reduction': True,
                    'sample_reduction_factor': 2,
                    'rhythmic_effects': ["Stutter"],
                    'arpeggiation': True,
                    'sequencer': True,
                    'sequence_pattern': 'Descending',
                    'effects_chain': ["Distortion"],
                    'effect_params': {'Distortion': {'gain': 30.0, 'threshold': 0.6}},
                    'voice_changer': False,
                    'pitch_shift_semitones': 0,
                    'algorithmic_composition': True,
                    'composition_type': "Rhythmic Pattern",
                    'synth_enabled': True,
                    'synth_notes': ['E2', 'G2', 'B2'],
                    'synth_scale': 'Minor',
                    'synth_waveform': 'Square',
                    'synth_attack': 0.05,
                    'synth_decay': 0.2,
                    'synth_sustain': 0.5,
                    'synth_release': 0.3
                }
            elif preset == "Factory Floor":
                params = {
                    'duration_type': 'Seconds',
                    'duration': 7,
                    'beats': 12,
                    'bpm': 110,
                    'noise_types': ["Pink Noise", "White Noise"],
                    'waveform_types': ["Sawtooth Wave"],
                    'lowcut': 80,
                    'highcut': 4000,
                    'order': 6,
                    'amplitude': 0.8,
                    'sample_rate': 44100,
                    'channels': "Stereo",
                    'fade_in': 0.3,
                    'fade_out': 0.3,
                    'panning': 0.4,
                    'bit_depth': 24,
                    'modulation': "Frequency Modulation",
                    'uploaded_file': None,
                    'reverse_audio': False,
                    'bitcrusher': False,
                    'bitcrusher_depth': 8,
                    'sample_reduction': True,
                    'sample_reduction_factor': 2,
                    'rhythmic_effects': ["Glitch"],
                    'arpeggiation': False,
                    'sequencer': True,
                    'sequence_pattern': 'Random',
                    'effects_chain': ["Reverb"],
                    'effect_params': {'Reverb': {'decay': 0.7}},
                    'voice_changer': False,
                    'pitch_shift_semitones': 0,
                    'algorithmic_composition': True,
                    'composition_type': "Ambient Soundscape",
                    'synth_enabled': True,
                    'synth_notes': ['A3', 'C4', 'E4'],
                    'synth_scale': 'Pentatonic',
                    'synth_waveform': 'Sawtooth',
                    'synth_attack': 0.02,
                    'synth_decay': 0.1,
                    'synth_sustain': 0.6,
                    'synth_release': 0.2
                }
            elif preset == "Electric Hum":
                params = {
                    'duration_type': 'Seconds',
                    'duration': 5,
                    'beats': 8,
                    'bpm': 100,
                    'noise_types': ["White Noise"],
                    'waveform_types': ["Sine Wave"],
                    'lowcut': 60,
                    'highcut': 120,
                    'order': 4,
                    'amplitude': 0.7,
                    'sample_rate': 48000,
                    'channels': "Mono",
                    'fade_in': 0.1,
                    'fade_out': 0.1,
                    'panning': 0.5,
                    'bit_depth': 16,
                    'modulation': "Amplitude Modulation",
                    'uploaded_file': None,
                    'reverse_audio': False,
                    'bitcrusher': False,
                    'bitcrusher_depth': 8,
                    'sample_reduction': False,
                    'sample_reduction_factor': 1,
                    'rhythmic_effects': [],
                    'arpeggiation': False,
                    'sequencer': False,
                    'sequence_pattern': 'Random',
                    'effects_chain': ["Tremolo"],
                    'effect_params': {'Tremolo': {'rate': 5.0, 'depth': 0.8}},
                    'voice_changer': False,
                    'pitch_shift_semitones': 0,
                    'algorithmic_composition': False,
                    'composition_type': None,
                    'synth_enabled': False,
                    'synth_notes': ['C4'],
                    'synth_scale': 'Major',
                    'synth_waveform': 'Sine',
                    'synth_attack': 0.01,
                    'synth_decay': 0.1,
                    'synth_sustain': 0.7,
                    'synth_release': 0.2
                }
            else:  # Custom
                params = None
            return params

        preset_params = set_preset(preset)

        if preset != "Custom" and preset_params is not None:
            # Set parameters from preset
            duration_type = preset_params.get('duration_type', 'Seconds')
            duration = preset_params['duration']
            beats = preset_params.get('beats', 8)
            bpm = preset_params.get('bpm', 120)
            noise_types = preset_params.get('noise_types', [])
            waveform_types = preset_params.get('waveform_types', [])
            lowcut = preset_params['lowcut']
            highcut = preset_params['highcut']
            order = preset_params['order']
            amplitude = preset_params['amplitude']
            sample_rate = preset_params['sample_rate']
            channels = preset_params['channels']
            fade_in = preset_params['fade_in']
            fade_out = preset_params['fade_out']
            panning = preset_params['panning']
            bit_depth = preset_params['bit_depth']
            modulation = preset_params.get('modulation', None)
            uploaded_file = preset_params.get('uploaded_file', None)
            reverse_audio = preset_params.get('reverse_audio', False)
            bitcrusher = preset_params.get('bitcrusher', False)
            bitcrusher_depth = preset_params.get('bitcrusher_depth', 8)
            sample_reduction = preset_params.get('sample_reduction', False)
            sample_reduction_factor = preset_params.get('sample_reduction_factor', 1)
            rhythmic_effects = preset_params.get('rhythmic_effects', [])
            arpeggiation = preset_params.get('arpeggiation', False)
            sequencer = preset_params.get('sequencer', False)
            sequence_pattern = preset_params.get('sequence_pattern', 'Random')
            effects_chain = preset_params.get('effects_chain', [])
            effect_params = preset_params.get('effect_params', {})
            voice_changer = preset_params.get('voice_changer', False)
            pitch_shift_semitones = preset_params.get('pitch_shift_semitones', 0)
            algorithmic_composition = preset_params.get('algorithmic_composition', False)
            composition_type = preset_params.get('composition_type', None)
            synth_enabled = preset_params.get('synth_enabled', False)
            synth_notes = preset_params.get('synth_notes', ['C4'])
            synth_scale = preset_params.get('synth_scale', 'Major')
            synth_waveform = preset_params.get('synth_waveform', 'Sine')
            synth_attack = preset_params.get('synth_attack', 0.01)
            synth_decay = preset_params.get('synth_decay', 0.1)
            synth_sustain = preset_params.get('synth_sustain', 0.7)
            synth_release = preset_params.get('synth_release', 0.2)
        else:
            # Custom parameters
            duration_type = st.selectbox(
                "Duration Type",
                ["Seconds", "Milliseconds", "Beats"],
                help="Choose how you want to specify the duration."
            )
            if duration_type == "Seconds":
                duration = st.slider(
                    "Duration (seconds)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    help="Select the duration of the noise in seconds."
                )
            elif duration_type == "Milliseconds":
                duration_ms = st.slider(
                    "Duration (milliseconds)",
                    min_value=100,
                    max_value=60000,
                    value=5000,
                    help="Select the duration of the noise in milliseconds."
                )
                duration = duration_ms / 1000.0  # Convert to seconds
            elif duration_type == "Beats":
                bpm = st.slider(
                    "BPM",
                    min_value=30,
                    max_value=300,
                    value=120,
                    help="Set the beats per minute."
                )
                beats = st.slider(
                    "Number of Beats",
                    min_value=1,
                    max_value=128,
                    value=8,
                    help="Set the number of beats."
                )
                duration = (60 / bpm) * beats  # Convert beats to seconds

            noise_options = ["White Noise", "Pink Noise", "Brown Noise", "Blue Noise", "Violet Noise", "Grey Noise"]
            noise_types = st.multiselect(
                "Noise Types",
                noise_options,
                help="Select one or more types of noise to include."
            )
            waveform_options = ["Sine Wave", "Square Wave", "Sawtooth Wave", "Triangle Wave"]
            waveform_types = st.multiselect(
                "Waveform Types",
                waveform_options,
                help="Select one or more waveforms to include."
            )
            lowcut = st.slider(
                "Low Cut Frequency (Hz)",
                min_value=20,
                max_value=10000,
                value=100,
                help="Set the low cut-off frequency for the bandpass filter."
            )
            highcut = st.slider(
                "High Cut Frequency (Hz)",
                min_value=1000,
                max_value=24000,
                value=5000,
                help="Set the high cut-off frequency for the bandpass filter."
            )
            order = st.slider(
                "Filter Order",
                min_value=1,
                max_value=10,
                value=6,
                help="Set the order of the bandpass filter."
            )
            amplitude = st.slider(
                "Amplitude",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                help="Set the amplitude (volume) of the generated noise."
            )
            sample_rate = st.selectbox(
                "Sample Rate (Hz)",
                [48000, 44100, 32000, 22050, 16000, 8000],
                index=0,
                help="Choose the sample rate for the audio."
            )
            channels = st.selectbox(
                "Channels",
                ["Mono", "Stereo"],
                help="Select mono or stereo output."
            )
            fade_in = st.slider(
                "Fade In Duration (seconds)",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                help="Set the duration for fade-in effect."
            )
            fade_out = st.slider(
                "Fade Out Duration (seconds)",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                help="Set the duration for fade-out effect."
            )
            panning = st.slider(
                "Panning (Stereo Only)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Adjust the panning for stereo output."
            )
            bit_depth = st.selectbox(
                "Bit Depth",
                [16, 24, 32],
                index=0,
                help="Select the bit depth for the audio."
            )
            st.markdown("### 🎵 Modulation")
            modulation = st.selectbox(
                "Modulation",
                [None, "Amplitude Modulation", "Frequency Modulation"],
                help="Choose a modulation effect to apply."
            )
            st.markdown("### 📁 Upload Audio")
            uploaded_file = st.file_uploader(
                "Upload an audio file to include",
                type=["wav", "mp3"],
                help="Include an external audio file in the mix."
            )
            st.markdown("### 🔄 Reverse Audio")
            reverse_audio = st.checkbox(
                "Enable Audio Reversal",
                help="Reverse the audio for a unique effect."
            )
            st.markdown("### 🛠️ Bitcrusher")
            bitcrusher = st.checkbox(
                "Enable Bitcrusher",
                help="Apply a bitcrusher effect to reduce audio fidelity."
            )
            if bitcrusher:
                bitcrusher_depth = st.slider(
                    "Bit Depth for Bitcrusher",
                    min_value=1,
                    max_value=16,
                    value=8,
                    help="Set the bit depth for the bitcrusher effect."
                )
            else:
                bitcrusher_depth = 16
            st.markdown("### 🔧 Sample Reduction")
            sample_reduction = st.checkbox(
                "Enable Sample Rate Reduction",
                help="Reduce the sample rate for a lo-fi effect."
            )
            if sample_reduction:
                sample_reduction_factor = st.slider(
                    "Reduction Factor",
                    min_value=1,
                    max_value=16,
                    value=1,
                    help="Set the factor by which to reduce the sample rate."
                )
            else:
                sample_reduction_factor = 1
            st.markdown("### 🎚️ Rhythmic Effects")
            rhythmic_effect_options = ["Stutter", "Glitch"]
            rhythmic_effects = st.multiselect(
                "Select Rhythmic Effects",
                rhythmic_effect_options,
                help="Choose rhythmic effects to apply."
            )
            st.markdown("### 🎹 Synthesizer")
            synth_enabled = st.checkbox(
                "Enable Synthesizer",
                help="Include synthesizer-generated tones."
            )
            if synth_enabled:
                note_options = ['C', 'C#', 'D', 'D#', 'E', 'F',
                                'F#', 'G', 'G#', 'A', 'A#', 'B']
                octave_options = [str(i) for i in range(1, 8)]
                selected_notes = st.multiselect(
                    "Notes",
                    [note + octave for octave in octave_options for note in note_options],
                    default=['C4'],
                    help="Select notes for the synthesizer."
                )
                synth_notes = selected_notes
                scale_options = ['Major', 'Minor', 'Pentatonic', 'Blues', 'Chromatic']
                synth_scale = st.selectbox(
                    "Scale",
                    scale_options,
                    help="Choose a scale for the synthesizer."
                )
                synth_waveform = st.selectbox(
                    "Waveform",
                    ["Sine", "Square", "Sawtooth", "Triangle"],
                    help="Select the waveform for the synthesizer."
                )
                st.markdown("**Envelope**")
                synth_attack = st.slider(
                    "Attack",
                    0.0,
                    1.0,
                    0.01,
                    help="Set the attack time for the envelope."
                )
                synth_decay = st.slider(
                    "Decay",
                    0.0,
                    1.0,
                    0.1,
                    help="Set the decay time for the envelope."
                )
                synth_sustain = st.slider(
                    "Sustain",
                    0.0,
                    1.0,
                    0.7,
                    help="Set the sustain level for the envelope."
                )
                synth_release = st.slider(
                    "Release",
                    0.0,
                    1.0,
                    0.2,
                    help="Set the release time for the envelope."
                )
            else:
                synth_notes = ['C4']
                synth_scale = 'Major'
                synth_waveform = 'Sine'
                synth_attack = 0.01
                synth_decay = 0.1
                synth_sustain = 0.7
                synth_release = 0.2
            st.markdown("### 🎹 Arpeggiation")
            arpeggiation = st.checkbox(
                "Enable Arpeggiation",
                help="Apply an arpeggiation effect to the notes."
            )
            st.markdown("### 🎛️ Sequencer")
            sequencer = st.checkbox(
                "Enable Sequencer",
                help="Sequence the generated tones."
            )
            if sequencer:
                sequence_patterns = ['Ascending', 'Descending', 'Random']
                sequence_pattern = st.selectbox(
                    "Sequence Pattern",
                    sequence_patterns,
                    help="Choose a pattern for the sequencer."
                )
            else:
                sequence_pattern = 'Random'
            st.markdown("### 🎚️ Effects Chain")
            effect_options = ["None", "Reverb", "Delay", "Distortion", "Tremolo", "Chorus", 
                              "Flanger", "Phaser", "Compression", "EQ", "Pitch Shifter", 
                              "High-pass Filter", "Low-pass Filter", "Vibrato", "Auto-pan"]
            effects_chain = []
            for i in range(1, 6):
                effect = st.selectbox(
                    f"Effect Slot {i}",
                    effect_options,
                    index=0,
                    key=f"effect_slot_{i}",
                    help="Select an effect to apply at this position in the chain."
                )
                if effect != "None":
                    effects_chain.append(effect)
            # Collect parameters for each effect
            effect_params = {}
            for effect in effects_chain:
                st.markdown(f"**{effect} Parameters**")
                params = {}
                if effect == "Reverb":
                    decay = st.slider(
                        "Reverb Decay",
                        0.1,
                        2.0,
                        0.5,
                        key=f"reverb_decay_{effect}",
                        help="Set the decay time for the reverb effect."
                    )
                    params['decay'] = decay
                elif effect == "Delay":
                    delay_time = st.slider(
                        "Delay Time (seconds)",
                        0.1,
                        1.0,
                        0.5,
                        key=f"delay_time_{effect}",
                        help="Set the delay time for the delay effect."
                    )
                    feedback = st.slider(
                        "Delay Feedback",
                        0.0,
                        1.0,
                        0.5,
                        key=f"delay_feedback_{effect}",
                        help="Set the feedback level for the delay effect."
                    )
                    params['delay_time'] = delay_time
                    params['feedback'] = feedback
                elif effect == "Distortion":
                    gain = st.slider(
                        "Distortion Gain",
                        1.0,
                        50.0,
                        20.0,
                        key=f"distortion_gain_{effect}",
                        help="Set the gain level for the distortion effect."
                    )
                    threshold = st.slider(
                        "Distortion Threshold",
                        0.0,
                        1.0,
                        0.5,
                        key=f"distortion_threshold_{effect}",
                        help="Set the threshold for the distortion effect."
                    )
                    params['gain'] = gain
                    params['threshold'] = threshold
                elif effect == "Tremolo":
                    rate = st.slider(
                        "Tremolo Rate (Hz)",
                        0.1,
                        20.0,
                        5.0,
                        key=f"tremolo_rate_{effect}",
                        help="Set the rate for the tremolo effect."
                    )
                    depth = st.slider(
                        "Tremolo Depth",
                        0.0,
                        1.0,
                        0.8,
                        key=f"tremolo_depth_{effect}",
                        help="Set the depth for the tremolo effect."
                    )
                    params['rate'] = rate
                    params['depth'] = depth
                elif effect == "Chorus":
                    rate = st.slider(
                        "Chorus Rate (Hz)",
                        0.1,
                        5.0,
                        1.5,
                        key=f"chorus_rate_{effect}",
                        help="Set the rate for the chorus effect."
                    )
                    depth = st.slider(
                        "Chorus Depth",
                        0.0,
                        1.0,
                        0.5,
                        key=f"chorus_depth_{effect}",
                        help="Set the depth for the chorus effect."
                    )
                    params['rate'] = rate
                    params['depth'] = depth
                elif effect == "Flanger":
                    rate = st.slider(
                        "Flanger Rate (Hz)",
                        0.1,
                        5.0,
                        0.5,
                        key=f"flanger_rate_{effect}",
                        help="Set the rate for the flanger effect."
                    )
                    depth = st.slider(
                        "Flanger Depth",
                        0.0,
                        1.0,
                        0.7,
                        key=f"flanger_depth_{effect}",
                        help="Set the depth for the flanger effect."
                    )
                    params['rate'] = rate
                    params['depth'] = depth
                elif effect == "Phaser":
                    rate = st.slider(
                        "Phaser Rate (Hz)",
                        0.1,
                        5.0,
                        0.5,
                        key=f"phaser_rate_{effect}",
                        help="Set the rate for the phaser effect."
                    )
                    depth = st.slider(
                        "Phaser Depth",
                        0.0,
                        1.0,
                        0.7,
                        key=f"phaser_depth_{effect}",
                        help="Set the depth for the phaser effect."
                    )
                    params['rate'] = rate
                    params['depth'] = depth
                elif effect == "Compression":
                    threshold = st.slider(
                        "Compressor Threshold",
                        0.0,
                        1.0,
                        0.5,
                        key=f"compressor_threshold_{effect}",
                        help="Set the threshold for the compressor."
                    )
                    ratio = st.slider(
                        "Compressor Ratio",
                        1.0,
                        20.0,
                        2.0,
                        key=f"compressor_ratio_{effect}",
                        help="Set the ratio for the compressor."
                    )
                    params['threshold'] = threshold
                    params['ratio'] = ratio
                elif effect == "EQ":
                    st.markdown("**Equalizer Bands**")
                    eq_bands = {}
                    for freq in ['Low', 'Mid', 'High']:
                        gain = st.slider(
                            f"{freq} Gain (dB)",
                            -12.0,
                            12.0,
                            0.0,
                            key=f"eq_{freq}_{effect}",
                            help=f"Set the gain for the {freq} frequencies."
                        )
                        eq_bands[freq.lower()] = gain
                    params['bands'] = eq_bands
                elif effect == "Pitch Shifter":
                    semitones = st.slider(
                        "Pitch Shift (semitones)",
                        -24,
                        24,
                        0,
                        key=f"pitch_shift_semitones_{effect}",
                        help="Set the number of semitones to shift the pitch."
                    )
                    params['semitones'] = semitones
                elif effect == "High-pass Filter":
                    cutoff = st.slider(
                        "High-pass Cutoff Frequency (Hz)",
                        20,
                        10000,
                        200,
                        key=f"highpass_cutoff_{effect}",
                        help="Set the cutoff frequency for the high-pass filter."
                    )
                    params['cutoff'] = cutoff
                elif effect == "Low-pass Filter":
                    cutoff = st.slider(
                        "Low-pass Cutoff Frequency (Hz)",
                        1000,
                        20000,
                        5000,
                        key=f"lowpass_cutoff_{effect}",
                        help="Set the cutoff frequency for the low-pass filter."
                    )
                    params['cutoff'] = cutoff
                elif effect == "Vibrato":
                    rate = st.slider(
                        "Vibrato Rate (Hz)",
                        0.1,
                        10.0,
                        5.0,
                        key=f"vibrato_rate_{effect}",
                        help="Set the rate for the vibrato effect."
                    )
                    depth = st.slider(
                        "Vibrato Depth",
                        0.0,
                        1.0,
                        0.5,
                        key=f"vibrato_depth_{effect}",
                        help="Set the depth for the vibrato effect."
                    )
                    params['rate'] = rate
                    params['depth'] = depth
                elif effect == "Auto-pan":
                    rate = st.slider(
                        "Auto-pan Rate (Hz)",
                        0.1,
                        10.0,
                        1.0,
                        key=f"autopan_rate_{effect}",
                        help="Set the rate for the auto-pan effect."
                    )
                    params['rate'] = rate
                effect_params[effect] = params

            # Voice Changer Controls
            st.markdown("### 🎤 Voice Changer")
            voice_changer = st.checkbox(
                "Enable Voice Changer (Pitch Shift)",
                help="Apply a pitch shift to a voice recording."
            )
            if voice_changer:
                voice_file = st.file_uploader(
                    "Upload your voice recording",
                    type=["wav", "mp3"],
                    help="Upload a voice recording to apply pitch shift."
                )
                pitch_shift_semitones = st.slider(
                    "Pitch Shift (semitones)",
                    min_value=-24,
                    max_value=24,
                    value=-5,
                    help="Set the number of semitones to shift the pitch."
                )
            else:
                voice_file = None
                pitch_shift_semitones = 0

            # Algorithmic Composition Controls
            st.markdown("### 🎼 Algorithmic Composition")
            algorithmic_composition = st.checkbox(
                "Enable Algorithmic Composition",
                help="Include algorithmically composed elements."
            )
            if algorithmic_composition:
                composition_options = ["Random Melody", "Ambient Soundscape", "Rhythmic Pattern"]
                composition_type = st.selectbox(
                    "Composition Type",
                    composition_options,
                    help="Choose a type of algorithmic composition."
                )
            else:
                composition_type = None

        # Collect waveform frequencies
        if waveform_types:
            st.markdown("### 🎶 Waveform Frequencies")
        waveform_frequencies = {}
        for waveform_type in waveform_types:
            frequency = st.slider(
                f"{waveform_type} Frequency (Hz)",
                min_value=20,
                max_value=20000,
                value=440,
                key=f"{waveform_type}_frequency_slider",
                help=f"Set the frequency for the {waveform_type.lower()}."
            )
            waveform_frequencies[waveform_type] = frequency

        # Preset Library Management
        st.markdown("### 💾 Preset Library")
        preset_name = st.text_input(
            "Preset Name",
            help="Enter a name to save the current settings as a preset."
        )
        if st.button("Save Preset"):
            current_params = {
                'duration_type': duration_type,
                'duration': duration,
                'beats': beats if 'beats' in locals() else None,
                'bpm': bpm if 'bpm' in locals() else None,
                'noise_types': noise_types,
                'waveform_types': waveform_types,
                'lowcut': lowcut,
                'highcut': highcut,
                'order': order,
                'amplitude': amplitude,
                'sample_rate': sample_rate,
                'channels': channels,
                'fade_in': fade_in,
                'fade_out': fade_out,
                'panning': panning,
                'bit_depth': bit_depth,
                'modulation': modulation,
                'uploaded_file': uploaded_file,
                'reverse_audio': reverse_audio,
                'bitcrusher': bitcrusher,
                'bitcrusher_depth': bitcrusher_depth,
                'sample_reduction': sample_reduction,
                'sample_reduction_factor': sample_reduction_factor,
                'rhythmic_effects': rhythmic_effects,
                'arpeggiation': arpeggiation,
                'sequencer': sequencer,
                'sequence_pattern': sequence_pattern,
                'effects_chain': effects_chain,
                'effect_params': effect_params,
                'voice_changer': voice_changer,
                'pitch_shift_semitones': pitch_shift_semitones,
                'algorithmic_composition': algorithmic_composition,
                'composition_type': composition_type,
                'synth_enabled': synth_enabled,
                'synth_notes': synth_notes,
                'synth_scale': synth_scale,
                'synth_waveform': synth_waveform,
                'synth_attack': synth_attack,
                'synth_decay': synth_decay,
                'synth_sustain': synth_sustain,
                'synth_release': synth_release
            }
            if preset_name:
                save_preset(current_params, preset_name)
                st.success(f"Preset '{preset_name}' saved!")
            else:
                st.error("Please enter a preset name.")

        available_presets = list_presets()
        if available_presets:
            load_preset_name = st.selectbox(
                "Load Preset",
                available_presets,
                help="Select a preset to load."
            )
            if st.button("Load Selected Preset"):
                loaded_params = load_preset(load_preset_name)
                st.success(f"Preset '{load_preset_name}' loaded!")
                st.experimental_rerun()

        st.markdown("---")

        # Initialize session state for temporary directory
        if 'temp_dir' not in st.session_state:
            st.session_state['temp_dir'] = tempfile.TemporaryDirectory()
            st.session_state['generated_files'] = []

        # Generate Noise Button
        if st.button("🎶 Generate Noise"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for sample_num in range(num_samples):
                status_text.text(f"Generating sample {sample_num + 1} of {num_samples}...")
                st.markdown(f"### Sample {sample_num + 1}")
                # Generate noise based on selection
                total_samples = int(duration * sample_rate)
                combined_data = np.zeros(total_samples, dtype=np.float32)

                # Introduce slight variations for each sample
                variation_factor = 0.05  # 5% variation

                # Vary frequency parameters
                frequency_variation = random.uniform(1 - variation_factor, 1 + variation_factor)
                varied_lowcut = lowcut * frequency_variation
                varied_highcut = highcut * frequency_variation

                # Vary amplitude
                amplitude_variation = random.uniform(1 - variation_factor, 1 + variation_factor)
                varied_amplitude = amplitude * amplitude_variation

                # Vary effect parameters
                varied_effect_params = {}
                for effect, params in effect_params.items():
                    varied_effect_params[effect] = {}
                    for param_name, value in params.items():
                        if isinstance(value, (int, float)):
                            variation = random.uniform(1 - variation_factor, 1 + variation_factor)
                            varied_effect_params[effect][param_name] = value * variation
                        else:
                            varied_effect_params[effect][param_name] = value

                # Generate noise types
                for noise_type in noise_types:
                    if noise_type == "White Noise":
                        data = generate_white_noise(duration, sample_rate)
                    elif noise_type == "Pink Noise":
                        data = generate_pink_noise(duration, sample_rate)
                    elif noise_type == "Brown Noise":
                        data = generate_brown_noise(duration, sample_rate)
                    elif noise_type == "Blue Noise":
                        data = generate_blue_noise(duration, sample_rate)
                    elif noise_type == "Violet Noise":
                        data = generate_violet_noise(duration, sample_rate)
                    elif noise_type == "Grey Noise":
                        data = generate_grey_noise(duration, sample_rate)
                    else:
                        data = np.zeros(total_samples, dtype=np.float32)

                    # Apply filter with varied parameters
                    data = apply_filter(data, varied_lowcut, varied_highcut, sample_rate, order)

                    # Normalize audio
                    data = data / np.max(np.abs(data) + 1e-7)
                    data = data.astype(np.float32)

                    # Adjust lengths if necessary
                    if len(data) != len(combined_data):
                        min_length = min(len(data), len(combined_data))
                        data = data[:min_length]
                        combined_data = combined_data[:min_length]

                    # Combine noises
                    combined_data += data

                # Generate waveform types
                for waveform_type in waveform_types:
                    frequency = waveform_frequencies[waveform_type]
                    frequency *= frequency_variation
                    data = generate_waveform(waveform_type, frequency, duration, sample_rate)
                    data = apply_filter(data, varied_lowcut, varied_highcut, sample_rate, order)
                    data = data / np.max(np.abs(data) + 1e-7)
                    data = data.astype(np.float32)

                    if len(data) != len(combined_data):
                        min_length = min(len(data), len(combined_data))
                        data = data[:min_length]
                        combined_data = combined_data[:min_length]

                    combined_data += data

                # Synthesizer
                if synth_enabled:
                    envelope = generate_envelope(total_samples, sample_rate, synth_attack, synth_decay, synth_sustain, synth_release)
                    synth_data = np.zeros(total_samples, dtype=np.float32)
                    for note in synth_notes:
                        note_freq = note_to_freq(note) * frequency_variation
                        tone = generate_synth_tone_from_freq(note_freq, duration, sample_rate, waveform=synth_waveform, envelope=envelope)
                        synth_data += tone
                    synth_data = synth_data / np.max(np.abs(synth_data) + 1e-7)
                    synth_data = synth_data.astype(np.float32)

                    if len(synth_data) != len(combined_data):
                        min_length = min(len(synth_data), len(combined_data))
                        synth_data = synth_data[:min_length]
                        combined_data = combined_data[:min_length]

                    combined_data += synth_data

                # Include uploaded audio file
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    try:
                        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True, duration=duration)
                        y = y[:total_samples]
                        y = y / np.max(np.abs(y) + 1e-7)
                        y = y.astype(np.float32)
                        if len(y) != len(combined_data):
                            min_length = min(len(y), len(combined_data))
                            y = y[:min_length]
                            combined_data = combined_data[:min_length]
                        combined_data += y
                    except Exception as e:
                        st.error(f"❌ Error loading uploaded audio: {e}")

                # Voice changer feature
                if voice_changer and voice_file is not None:
                    voice_bytes = voice_file.read()
                    try:
                        y, sr = librosa.load(io.BytesIO(voice_bytes), sr=sample_rate, mono=True)
                        y_shifted = pitch_shift_audio(y, sr, n_steps=pitch_shift_semitones)
                        if len(y_shifted) > total_samples:
                            y_shifted = y_shifted[:total_samples]
                        else:
                            y_shifted = np.pad(y_shifted, (0, total_samples - len(y_shifted)), 'constant')
                        y_shifted = y_shifted / np.max(np.abs(y_shifted) + 1e-7)
                        y_shifted = y_shifted.astype(np.float32)

                        if len(y_shifted) != len(combined_data):
                            min_length = min(len(y_shifted), len(combined_data))
                            y_shifted = y_shifted[:min_length]
                            combined_data = combined_data[:min_length]

                        combined_data += y_shifted
                    except Exception as e:
                        st.error(f"❌ Error applying voice changer: {e}")

                # Include algorithmic composition
                if algorithmic_composition and composition_type is not None:
                    data = generate_algorithmic_composition(duration, sample_rate, composition_type)
                    data = data / np.max(np.abs(data) + 1e-7)
                    data = data.astype(np.float32)

                    if len(data) != len(combined_data):
                        min_length = min(len(data), len(combined_data))
                        data = data[:min_length]
                        combined_data = combined_data[:min_length]

                    combined_data += data

                # Normalize combined data
                combined_data = combined_data / np.max(np.abs(combined_data) + 1e-7)

                # Copy for plotting
                plot_data = combined_data.copy()

                # Apply varied amplitude
                combined_data *= varied_amplitude
                plot_data *= varied_amplitude

                # Apply modulation
                if modulation == "Amplitude Modulation":
                    combined_data = apply_amplitude_modulation(combined_data, sample_rate)
                    plot_data = apply_amplitude_modulation(plot_data, sample_rate)
                elif modulation == "Frequency Modulation":
                    combined_data = apply_frequency_modulation(combined_data, sample_rate)
                    plot_data = apply_frequency_modulation(plot_data, sample_rate)

                # Apply fade in/out
                combined_data = apply_fade(combined_data, sample_rate, fade_in, fade_out)
                plot_data = apply_fade(plot_data, sample_rate, fade_in, fade_out)

                # Apply reverse
                if reverse_audio:
                    combined_data = apply_reverse(combined_data)
                    plot_data = apply_reverse(plot_data)

                # Apply bitcrusher
                if bitcrusher:
                    combined_data = apply_bitcrusher(combined_data, bitcrusher_depth)
                    plot_data = apply_bitcrusher(plot_data, bitcrusher_depth)

                # Apply sample rate reduction
                if sample_reduction and sample_reduction_factor > 1:
                    combined_data = apply_sample_reduction(combined_data, sample_reduction_factor)
                    plot_data = apply_sample_reduction(plot_data, sample_reduction_factor)

                # Apply rhythmic effects
                for effect in rhythmic_effects:
                    if effect == "Stutter":
                        combined_data = apply_stutter(combined_data, sample_rate)
                        plot_data = apply_stutter(plot_data, sample_rate)
                    elif effect == "Glitch":
                        combined_data = apply_glitch(combined_data, sample_rate)
                        plot_data = apply_glitch(plot_data, sample_rate)

                # Apply arpeggiation
                if arpeggiation:
                    combined_data = apply_arpeggiation(combined_data, sample_rate, pattern=sequence_pattern)
                    plot_data = apply_arpeggiation(plot_data, sample_rate, pattern=sequence_pattern)

                # Apply sequencer
                if sequencer:
                    combined_data = apply_sequencer(combined_data, sample_rate, pattern=sequence_pattern)
                    plot_data = apply_sequencer(plot_data, sample_rate, pattern=sequence_pattern)

                # Apply effects in chain order
                for effect in effects_chain:
                    params = varied_effect_params.get(effect, {})
                    if effect == "Reverb":
                        combined_data = apply_reverb(combined_data, sample_rate, decay=params.get('decay', 0.5))
                        plot_data = apply_reverb(plot_data, sample_rate, decay=params.get('decay', 0.5))
                    elif effect == "Delay":
                        combined_data = apply_delay(combined_data, sample_rate, delay_time=params.get('delay_time', 0.5), feedback=params.get('feedback', 0.5))
                        plot_data = apply_delay(plot_data, sample_rate, delay_time=params.get('delay_time', 0.5), feedback=params.get('feedback', 0.5))
                    elif effect == "Distortion":
                        combined_data = apply_distortion(combined_data, gain=params.get('gain', 20), threshold=params.get('threshold', 0.5))
                        plot_data = apply_distortion(plot_data, gain=params.get('gain', 20), threshold=params.get('threshold', 0.5))
                    elif effect == "Tremolo":
                        combined_data = apply_tremolo(combined_data, sample_rate, rate=params.get('rate', 5.0), depth=params.get('depth', 0.8))
                        plot_data = apply_tremolo(plot_data, sample_rate, rate=params.get('rate', 5.0), depth=params.get('depth', 0.8))
                    elif effect == "Chorus":
                        combined_data = apply_chorus(combined_data, sample_rate, rate=params.get('rate', 1.5), depth=params.get('depth', 0.5))
                        plot_data = apply_chorus(plot_data, sample_rate, rate=params.get('rate', 1.5), depth=params.get('depth', 0.5))
                    elif effect == "Flanger":
                        combined_data = apply_flanger(combined_data, sample_rate, rate=params.get('rate', 0.5), depth=params.get('depth', 0.7))
                        plot_data = apply_flanger(plot_data, sample_rate, rate=params.get('rate', 0.5), depth=params.get('depth', 0.7))
                    elif effect == "Phaser":
                        combined_data = apply_phaser(combined_data, sample_rate, rate=params.get('rate', 0.5), depth=params.get('depth', 0.7))
                        plot_data = apply_phaser(plot_data, sample_rate, rate=params.get('rate', 0.5), depth=params.get('depth', 0.7))
                    elif effect == "Compression":
                        combined_data = apply_compression(combined_data, threshold=params.get('threshold', 0.5), ratio=params.get('ratio', 2.0))
                        plot_data = apply_compression(plot_data, threshold=params.get('threshold', 0.5), ratio=params.get('ratio', 2.0))
                    elif effect == "EQ":
                        combined_data = apply_eq(combined_data, sample_rate, params.get('bands', {'low': 0.0, 'mid': 0.0, 'high': 0.0}))
                        plot_data = apply_eq(plot_data, sample_rate, params.get('bands', {'low': 0.0, 'mid': 0.0, 'high': 0.0}))
                    elif effect == "Pitch Shifter":
                        combined_data = apply_pitch_shift(combined_data, sample_rate, params.get('semitones', 0))
                        plot_data = apply_pitch_shift(plot_data, sample_rate, params.get('semitones', 0))
                    elif effect == "High-pass Filter":
                        combined_data = apply_highpass_filter(combined_data, sample_rate, params.get('cutoff', 200))
                        plot_data = apply_highpass_filter(plot_data, sample_rate, params.get('cutoff', 200))
                    elif effect == "Low-pass Filter":
                        combined_data = apply_lowpass_filter(combined_data, sample_rate, params.get('cutoff', 5000))
                        plot_data = apply_lowpass_filter(plot_data, sample_rate, params.get('cutoff', 5000))
                    elif effect == "Vibrato":
                        combined_data = apply_vibrato(combined_data, sample_rate, rate=params.get('rate', 5.0), depth=params.get('depth', 0.5))
                        plot_data = apply_vibrato(plot_data, sample_rate, rate=params.get('rate', 5.0), depth=params.get('depth', 0.5))
                    elif effect == "Auto-pan":
                        combined_data = apply_autopan(combined_data, sample_rate, rate=params.get('rate', 1.0))
                        plot_data = apply_autopan(plot_data, sample_rate, rate=params.get('rate', 1.0))

                # Adjust bit depth
                combined_data = adjust_bit_depth(combined_data, bit_depth)
                plot_data = adjust_bit_depth(plot_data, bit_depth)

                # Apply panning for stereo output
                if channels == "Stereo":
                    combined_data = pan_stereo(combined_data, panning)
                    plot_data = pan_stereo(plot_data, panning)

                # Normalize audio again after all processing
                combined_data = combined_data / np.max(np.abs(combined_data) + 1e-7)
                plot_data = plot_data / np.max(np.abs(plot_data) + 1e-7)

                # Convert to int16 for saving
                combined_data = (combined_data * 32767).astype(np.int16)

                # Generate a unique filename
                filename = f"industrial_noise_{sample_num + 1}.wav"
                filepath = os.path.join(st.session_state['temp_dir'].name, filename)

                # Save the audio file
                write(filepath, sample_rate, combined_data)
                st.session_state['generated_files'].append(filepath)

                # Display audio player
                st.audio(filepath)

                # Create interactive waveform plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=plot_data, mode='lines', name='Waveform'))
                fig.update_layout(
                    title='Interactive Waveform',
                    xaxis_title='Sample',
                    yaxis_title='Amplitude',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Create spectrogram
                try:
                    D = librosa.stft(plot_data)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax)
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax.set_title('Spectrogram')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Error generating spectrogram: {e}")

                # Update progress
                progress_bar.progress((sample_num + 1) / num_samples)

            status_text.text("All samples generated!")

        # Download all generated files as a zip
        if st.session_state['generated_files']:
            zip_filename = "industrial_noise_samples.zip"
            zip_path = os.path.join(st.session_state['temp_dir'].name, zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in st.session_state['generated_files']:
                    zipf.write(file, os.path.basename(file))
            
            with open(zip_path, "rb") as f:
                btn = st.download_button(
                    label="Download All Samples",
                    data=f.read(),
                    file_name=zip_filename,
                    mime="application/zip"
                )

    # -------------------- Tab 2: Audio Recorder and Playback --------------------
    with tabs[1]:
        st.header("🎙️ Audio Recorder and Playback")

        # Stylish description with Apple-inspired design
        html_description = """
        <div style="background-color:#f5f5f7;padding:20px;border-radius:20px;box-shadow:0 4px 8px rgba(0, 0, 0, 0.1);">
            <h1 style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#1d1d1f;">Welcome to the Audio Recorder</h1>
            <p style="font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;color:#6e6e73;font-size:16px;">Upload, analyze, and enhance your audio with a simple, intuitive interface.</p>
        </div>
        """
        html(html_description, height=180)

        # Step 1: Upload Audio
        st.write("### Step 1: Upload your audio")
        audio_file = st.file_uploader("Please upload your audio file:", type=["wav", "mp3"])

        if audio_file is not None:
            try:
                # Load the uploaded file
                with wave.open(audio_file, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    n_frames = wav_file.getnframes()
                    audio_signal = wav_file.readframes(n_frames)
                    audio_signal = np.frombuffer(audio_signal, dtype=np.int16)
                    if n_channels > 1:
                        audio_signal = audio_signal.reshape(-1, n_channels)
                        audio_signal = audio_signal.mean(axis=1)  # Convert to mono by averaging channels
                    audio_signal = audio_signal.astype(np.float32)
                    audio_signal /= np.max(np.abs(audio_signal) + 1e-7)  # Normalize
            except wave.Error:
                # If the file is not a WAV file, try loading with librosa
                try:
                    audio_signal, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                    audio_signal = audio_signal.astype(np.float32)
                    audio_signal /= np.max(np.abs(audio_signal) + 1e-7)  # Normalize
                except Exception as e:
                    st.error(f"❌ Error loading audio file: {e}")
                    audio_signal = None
                    sample_rate = None

            if audio_signal is not None:
                # Step 2: Playback and Monitoring
                st.write("### Step 2: Play back your recording")
                st.audio(audio_file, format='audio/wav')

                # Step 3: Save Audio
                st.write("### Step 3: Save or analyze your recording")
                save_option = st.button("Save Recording")
                if save_option:
                    with open("recorded_audio.wav", "wb") as f:
                        f.write(audio_file.read())
                    st.success("✅ Audio recording saved as 'recorded_audio.wav'")

                # Step 4: Analyze Audio
                analyze_button = st.button("Analyze Recording")
                if analyze_button:
                    try:
                        duration = len(audio_signal) / sample_rate
                        st.write("#### Audio Analysis")
                        st.write(f"- **Duration:** {duration:.2f} seconds")
                        st.write(f"- **Sample Rate:** {sample_rate} Hz")
                        st.write(f"- **Channels:** {'Stereo' if n_channels > 1 else 'Mono'}")
                        st.write(f"- **Sample Width:** {sampwidth} bytes" if 'sampwidth' in locals() else "- **Sample Width:** N/A")

                        # Visualization
                        st.write("#### Waveform")
                        fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
                        ax_wave.plot(audio_signal, color='cyan')
                        ax_wave.set_title('Waveform')
                        ax_wave.set_xlabel('Sample')
                        ax_wave.set_ylabel('Amplitude')
                        ax_wave.set_facecolor('#1a1a1a')
                        ax_wave.tick_params(colors='white')
                        fig_wave.patch.set_facecolor('#1a1a1a')
                        st.pyplot(fig_wave)

                        # Spectrogram
                        st.write("#### Spectrogram")
                        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                        S = librosa.stft(audio_signal)
                        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                        librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax_spec, cmap='magma')
                        ax_spec.set_title('Spectrogram')
                        ax_spec.set_facecolor('#1a1a1a')
                        ax_spec.tick_params(colors='white')
                        fig_spec.colorbar(img:=librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax_spec, cmap='magma'), ax=ax_spec, format="%+2.0f dB")
                        fig_spec.patch.set_facecolor('#1a1a1a')
                        st.pyplot(fig_spec)
                    except Exception as e:
                        st.error(f"❌ Error analyzing audio: {e}")

                # Step 5: Apply Effects
                st.write("### Step 4: Apply effects to your recording")
                effect_option = st.selectbox(
                    "Choose an effect to apply:",
                    ["None", "Low Pass Filter", "High Pass Filter", "Amplify", "Echo", "Reverb", "Distortion", "FFT Filter", "Envelope Modulation"],
                    help="Select an audio effect to enhance your recording."
                )
                if effect_option != "None":
                    try:
                        processed_signal = apply_effect(effect_option, audio_signal, sample_rate)

                        st.write(f"### Effect Applied: {effect_option}")
                        st.write("#### Processed Waveform")
                        fig_proc_wave, ax_proc_wave = plt.subplots(figsize=(10, 2))
                        ax_proc_wave.plot(processed_signal, color='orange')
                        ax_proc_wave.set_title('Processed Waveform')
                        ax_proc_wave.set_xlabel('Sample')
                        ax_proc_wave.set_ylabel('Amplitude')
                        ax_proc_wave.set_facecolor('#1a1a1a')
                        ax_proc_wave.tick_params(colors='white')
                        fig_proc_wave.patch.set_facecolor('#1a1a1a')
                        st.pyplot(fig_proc_wave)

                        st.write("#### Processed Spectrogram")
                        fig_proc_spec, ax_proc_spec = plt.subplots(figsize=(10, 4))
                        S_proc = librosa.stft(processed_signal)
                        S_proc_db = librosa.amplitude_to_db(np.abs(S_proc), ref=np.max)
                        librosa.display.specshow(S_proc_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax_proc_spec, cmap='magma')
                        ax_proc_spec.set_title('Processed Spectrogram')
                        ax_proc_spec.set_facecolor('#1a1a1a')
                        ax_proc_spec.tick_params(colors='white')
                        fig_proc_spec.colorbar(img:=librosa.display.specshow(S_proc_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax_proc_spec, cmap='magma'), ax=ax_proc_spec, format="%+2.0f dB")
                        fig_proc_spec.patch.set_facecolor('#1a1a1a')
                        st.pyplot(fig_proc_spec)

                        # Save processed audio
                        if st.button("Save Processed Audio"):
                            # Convert to int16
                            processed_int16 = np.int16(processed_signal / np.max(np.abs(processed_signal)) * 32767)
                            with wave.open("processed_audio.wav", "wb") as processed_file:
                                processed_file.setnchannels(1)
                                processed_file.setsampwidth(2)  # 16-bit
                                processed_file.setframerate(sample_rate)
                                processed_file.writeframes(processed_int16.tobytes())
                            st.success("✅ Processed audio saved as 'processed_audio.wav'")
                    except Exception as e:
                        st.error(f"❌ Error applying effect: {e}")

    # Run the application
    if __name__ == "__main__":
        main()
