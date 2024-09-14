# Industrial Noise Generator Pro Max with Variations

import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import scipy.signal as signal
import matplotlib.pyplot as plt
import io
import librosa
import librosa.display
import random
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# For file management
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="Industrial Noise Generator Pro Max with Variations",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance appearance
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: white;
    }
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
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎹 Industrial Noise Generator Pro Max with Variations")
st.markdown("""
Generate multiple industrial noise samples with advanced features, including BPM customization, note and scale selection, and a built-in synthesizer. Customize parameters to create unique sounds, and generate variations automatically!
""")

# Sidebar for parameters
st.sidebar.header("🎛️ Controls")

# Number of samples to generate
st.sidebar.subheader("🔢 Sample Generation")
num_samples = st.sidebar.number_input("Number of Samples to Generate", min_value=1, max_value=100, value=1)

# Presets
st.sidebar.subheader("🎚️ Presets")
preset_options = ["Default", "Heavy Machinery", "Factory Floor", "Electric Hum", "Custom"]
preset = st.sidebar.selectbox("Choose a Preset", preset_options)

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
            'effects': [],
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
    # Additional presets can be defined similarly...
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
    effects = preset_params.get('effects', [])
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
    duration_type = st.sidebar.selectbox("Duration Type", ["Seconds", "Milliseconds", "Beats"])
    if duration_type == "Seconds":
        duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=5)
    elif duration_type == "Milliseconds":
        duration_ms = st.sidebar.slider("Duration (milliseconds)", min_value=100, max_value=60000, value=5000)
        duration = duration_ms / 1000.0  # Convert to seconds
    elif duration_type == "Beats":
        bpm = st.sidebar.slider("BPM", min_value=30, max_value=300, value=120)
        beats = st.sidebar.slider("Number of Beats", min_value=1, max_value=128, value=8)
        duration = (60 / bpm) * beats  # Convert beats to seconds

    noise_options = ["White Noise", "Pink Noise", "Brown Noise", "Blue Noise", "Violet Noise", "Grey Noise"]
    noise_types = st.sidebar.multiselect("Noise Types", noise_options)
    waveform_options = ["Sine Wave", "Square Wave", "Sawtooth Wave", "Triangle Wave"]
    waveform_types = st.sidebar.multiselect("Waveform Types", waveform_options)
    lowcut = st.sidebar.slider("Low Cut Frequency (Hz)", min_value=20, max_value=10000, value=100)
    highcut = st.sidebar.slider("High Cut Frequency (Hz)", min_value=1000, max_value=24000, value=5000)
    order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=6)
    amplitude = st.sidebar.slider("Amplitude", min_value=0.0, max_value=1.0, value=1.0)
    sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [48000, 44100, 32000, 22050, 16000, 8000], index=0)
    channels = st.sidebar.selectbox("Channels", ["Mono", "Stereo"])
    fade_in = st.sidebar.slider("Fade In Duration (seconds)", min_value=0.0, max_value=5.0, value=0.0)
    fade_out = st.sidebar.slider("Fade Out Duration (seconds)", min_value=0.0, max_value=5.0, value=0.0)
    panning = st.sidebar.slider("Panning (Stereo Only)", min_value=0.0, max_value=1.0, value=0.5)
    bit_depth = st.sidebar.selectbox("Bit Depth", [16, 24, 32], index=0)
    st.sidebar.subheader("🎵 Modulation")
    modulation = st.sidebar.selectbox("Modulation", [None, "Amplitude Modulation", "Frequency Modulation"])
    st.sidebar.subheader("📁 Upload Audio")
    uploaded_file = st.sidebar.file_uploader("Upload an audio file to include", type=["wav", "mp3"])
    st.sidebar.subheader("🔄 Reverse Audio")
    reverse_audio = st.sidebar.checkbox("Enable Audio Reversal")
    st.sidebar.subheader("🛠️ Bitcrusher")
    bitcrusher = st.sidebar.checkbox("Enable Bitcrusher")
    if bitcrusher:
        bitcrusher_depth = st.sidebar.slider("Bit Depth for Bitcrusher", min_value=1, max_value=16, value=8)
    else:
        bitcrusher_depth = 16
    st.sidebar.subheader("🔧 Sample Reduction")
    sample_reduction = st.sidebar.checkbox("Enable Sample Rate Reduction")
    if sample_reduction:
        sample_reduction_factor = st.sidebar.slider("Reduction Factor", min_value=1, max_value=16, value=1)
    else:
        sample_reduction_factor = 1
    st.sidebar.subheader("🎚️ Rhythmic Effects")
    rhythmic_effect_options = ["Stutter", "Glitch"]
    rhythmic_effects = st.sidebar.multiselect("Select Rhythmic Effects", rhythmic_effect_options)
    st.sidebar.subheader("🎹 Synthesizer")
    synth_enabled = st.sidebar.checkbox("Enable Synthesizer")
    if synth_enabled:
        note_options = ['C', 'C#', 'D', 'D#', 'E', 'F',
                        'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave_options = [str(i) for i in range(1, 8)]
        selected_notes = st.sidebar.multiselect("Notes", [note + octave for octave in octave_options for note in note_options], default=['C4'])
        synth_notes = selected_notes
        scale_options = ['Major', 'Minor', 'Pentatonic', 'Blues', 'Chromatic']
        synth_scale = st.sidebar.selectbox("Scale", scale_options)
        synth_waveform = st.sidebar.selectbox("Waveform", ["Sine", "Square", "Sawtooth", "Triangle"])
        st.sidebar.markdown("**Envelope**")
        synth_attack = st.sidebar.slider("Attack", 0.0, 1.0, 0.01)
        synth_decay = st.sidebar.slider("Decay", 0.0, 1.0, 0.1)
        synth_sustain = st.sidebar.slider("Sustain", 0.0, 1.0, 0.7)
        synth_release = st.sidebar.slider("Release", 0.0, 1.0, 0.2)
    else:
        synth_notes = ['C4']
        synth_scale = 'Major'
        synth_waveform = 'Sine'
        synth_attack = 0.01
        synth_decay = 0.1
        synth_sustain = 0.7
        synth_release = 0.2
    st.sidebar.subheader("🎹 Arpeggiation")
    arpeggiation = st.sidebar.checkbox("Enable Arpeggiation")
    st.sidebar.subheader("🎛️ Sequencer")
    sequencer = st.sidebar.checkbox("Enable Sequencer")
    if sequencer:
        sequence_patterns = ['Ascending', 'Descending', 'Random']
        sequence_pattern = st.sidebar.selectbox("Sequence Pattern", sequence_patterns)
    else:
        sequence_pattern = 'Random'
    st.sidebar.subheader("🎚️ Effects")
    effect_options = ["Reverb", "Delay", "Distortion", "Tremolo"]
    effects = st.sidebar.multiselect("Select Effects", effect_options)
    effect_params = {}
    for effect in effects:
        st.sidebar.markdown(f"**{effect} Parameters**")
        if effect == "Reverb":
            decay = st.sidebar.slider("Reverb Decay", 0.1, 2.0, 0.5, key=f"reverb_decay_{effect}")
            effect_params['Reverb'] = {'decay': decay}
        elif effect == "Delay":
            delay_time = st.sidebar.slider("Delay Time (seconds)", 0.1, 1.0, 0.5, key=f"delay_time_{effect}")
            feedback = st.sidebar.slider("Delay Feedback", 0.0, 1.0, 0.5, key=f"delay_feedback_{effect}")
            effect_params['Delay'] = {'delay_time': delay_time, 'feedback': feedback}
        elif effect == "Distortion":
            gain = st.sidebar.slider("Distortion Gain", 1.0, 50.0, 20.0, key=f"distortion_gain_{effect}")
            threshold = st.sidebar.slider("Distortion Threshold", 0.0, 1.0, 0.5, key=f"distortion_threshold_{effect}")
            effect_params['Distortion'] = {'gain': gain, 'threshold': threshold}
        elif effect == "Tremolo":
            rate = st.sidebar.slider("Tremolo Rate (Hz)", 0.1, 20.0, 5.0, key=f"tremolo_rate_{effect}")
            depth = st.sidebar.slider("Tremolo Depth", 0.0, 1.0, 0.8, key=f"tremolo_depth_{effect}")
            effect_params['Tremolo'] = {'rate': rate, 'depth': depth}
    st.sidebar.subheader("🎤 Voice Changer")
    voice_changer = st.sidebar.checkbox("Enable Voice Changer (Pitch Shift)")
    if voice_changer:
        voice_file = st.sidebar.file_uploader("Upload your voice recording", type=["wav", "mp3"])
        pitch_shift_semitones = st.sidebar.slider("Pitch Shift (semitones)", min_value=-24, max_value=24, value=-5)
    else:
        voice_file = None
        pitch_shift_semitones = 0
    st.sidebar.subheader("🎼 Algorithmic Composition")
    algorithmic_composition = st.sidebar.checkbox("Enable Algorithmic Composition")
    if algorithmic_composition:
        composition_options = ["Random Melody", "Ambient Soundscape", "Rhythmic Pattern"]
        composition_type = st.sidebar.selectbox("Composition Type", composition_options)
    else:
        composition_type = None

# Collect waveform frequencies outside the main function
waveform_frequencies = {}
for waveform_type in waveform_types:
    frequency = st.sidebar.slider(
        f"{waveform_type} Frequency (Hz)",
        min_value=20,
        max_value=20000,
        value=440,
        key=f"{waveform_type}_frequency_slider"
    )
    waveform_frequencies[waveform_type] = frequency

# Functions to generate noise and effects (same as your original code)
# ... [Insert all your function definitions here without changes] ...

# File library functions
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

# Main function
def main():
    # File library management
    st.sidebar.subheader("💾 Preset Library")
    preset_name = st.sidebar.text_input("Preset Name")
    if st.sidebar.button("Save Preset"):
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
            'reverse_audio': reverse_audio,
            'bitcrusher': bitcrusher,
            'bitcrusher_depth': bitcrusher_depth,
            'sample_reduction': sample_reduction,
            'sample_reduction_factor': sample_reduction_factor,
            'rhythmic_effects': rhythmic_effects,
            'arpeggiation': arpeggiation,
            'sequencer': sequencer,
            'sequence_pattern': sequence_pattern,
            'effects': effects,
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
        save_preset(current_params, preset_name)
        st.sidebar.success(f"Preset '{preset_name}' saved!")

    available_presets = list_presets()
    if available_presets:
        load_preset_name = st.sidebar.selectbox("Load Preset", available_presets)
        if st.sidebar.button("Load Selected Preset"):
            loaded_params = load_preset(load_preset_name)
            st.sidebar.success(f"Preset '{load_preset_name}' loaded!")
            # Update parameters with loaded preset
            st.experimental_rerun()

    if st.button("🎶 Generate Noise"):
        for sample_num in range(num_samples):
            st.markdown(f"### Sample {sample_num + 1}")
            # Generate noise based on selection
            total_samples = int(duration * sample_rate)
            combined_data = np.zeros(total_samples)

            # Introduce slight variations for each sample
            variation_factor = 0.05  # 5% variation
            varied_params = {}

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
                    data = np.zeros(total_samples)

                # Apply filter with varied parameters
                data = apply_filter(data, varied_lowcut, varied_highcut, sample_rate, order)

                # Normalize audio
                data = data / np.max(np.abs(data) + 1e-7)

                # Combine noises
                combined_data += data

            # Generate waveform types
            for waveform_type in waveform_types:
                frequency = waveform_frequencies[waveform_type]
                # Apply variation to frequency
                frequency *= frequency_variation
                data = generate_waveform(waveform_type, frequency, duration, sample_rate)
                # Apply filter with varied parameters
                data = apply_filter(data, varied_lowcut, varied_highcut, sample_rate, order)
                # Normalize audio
                data = data / np.max(np.abs(data) + 1e-7)
                # Combine waveforms
                combined_data += data

            # Synthesizer
            if synth_enabled:
                envelope = generate_envelope(total_samples, sample_rate, synth_attack, synth_decay, synth_sustain, synth_release)
                synth_data = np.zeros(total_samples)
                for note in synth_notes:
                    # Apply variation to note frequency
                    note_freq = note_to_freq(note) * frequency_variation
                    # Generate tone with varied frequency
                    tone = generate_synth_tone_from_freq(note_freq, duration, sample_rate, waveform=synth_waveform, envelope=envelope)
                    synth_data += tone
                synth_data = synth_data / np.max(np.abs(synth_data) + 1e-7)
                combined_data += synth_data

            # Include uploaded audio file
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                # Load the uploaded file
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True, duration=duration)
                y = y[:total_samples]  # Ensure length matches
                y = y / np.max(np.abs(y) + 1e-7)  # Normalize
                combined_data += y

            # Voice changer feature
            if voice_changer and voice_file is not None:
                voice_bytes = voice_file.read()
                # Load the voice file
                y, sr = librosa.load(io.BytesIO(voice_bytes), sr=sample_rate, mono=True)
                # Pitch shift by specified semitones
                y_shifted = pitch_shift_audio(y, sr, n_steps=pitch_shift_semitones)
                # Ensure length matches
                if len(y_shifted) > total_samples:
                    y_shifted = y_shifted[:total_samples]
                else:
                    y_shifted = np.pad(y_shifted, (0, total_samples - len(y_shifted)), 'constant')
                y_shifted = y_shifted / np.max(np.abs(y_shifted) + 1e-7)
                combined_data += y_shifted

            # Include algorithmic composition
            if algorithmic_composition and composition_type is not None:
                data = generate_algorithmic_composition(duration, sample_rate, composition_type)
                data = data / np.max(np.abs(data) + 1e-7)
                combined_data += data

            # Normalize combined data
            combined_data = combined_data / np.max(np.abs(combined_data) + 1e-7)

            # Make a copy for plotting before converting to integer types
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

            # Apply other effects with varied parameters
            for effect in effects:
                params = varied_effect_params.get(effect, {})
                if effect == "Reverb":
                    combined_data = apply_reverb(combined_data, sample_rate, decay=params['decay'])
                    plot_data = apply_reverb(plot_data, sample_rate, decay=params['decay'])
                elif effect == "Delay":
                    combined_data = apply_delay(combined_data, sample_rate, delay_time=params['delay_time'], feedback=params['feedback'])
                    plot_data = apply_delay(plot_data, sample_rate, delay_time=params['delay_time'], feedback=params['feedback'])
                elif effect == "Distortion":
                    combined_data = apply_distortion(combined_data, gain=params['gain'], threshold=params['threshold'])
                    plot_data = apply_distortion(plot_data, gain=params['gain'], threshold=params['threshold'])
                elif effect == "Tremolo":
                    combined_data = apply_tremolo(combined_data, sample_rate, rate=params['rate'], depth=params['depth'])
                    plot_data = apply_tremolo(plot_data, sample_rate, rate=params['rate'], depth=params['depth'])

            # Adjust bit depth on combined_data (not on plot_data)
            combined_data = adjust_bit_depth(combined_data, bit_depth)

            # Handle stereo or mono for combined_data
            if channels == "Stereo":
                combined_data = pan_stereo(combined_data, panning)
            else:
                combined_data = combined_data.reshape(-1, 1)

            # Handle stereo or mono for plot_data
            if channels == "Stereo":
                plot_data = pan_stereo(plot_data, panning)
            else:
                plot_data = plot_data.reshape(-1, 1)

            # Convert combined_data to proper dtype for saving
            if bit_depth == 16:
                dtype = np.int16
                max_int = np.iinfo(dtype).max
                combined_data = combined_data * max_int
                combined_data = combined_data.astype(dtype)
            elif bit_depth == 24:
                # 24-bit WAV files are supported by soundfile library
                dtype = 'int24'
                combined_data = combined_data * (2**23 - 1)
                combined_data = combined_data.astype(np.int32)
            else:  # 32-bit
                dtype = np.float32
                combined_data = combined_data.astype(dtype)

            # Save audio to buffer
            buffer = io.BytesIO()
            if bit_depth == 24:
                # Use soundfile to write 24-bit audio
                sf.write(buffer, combined_data, sample_rate, subtype='PCM_24')
            else:
                write(buffer, sample_rate, combined_data)
            buffer.seek(0)

            # Play audio
            st.audio(buffer, format='audio/wav')

            # Provide download button
            st.download_button(label=f"💾 Download Sample {sample_num + 1}", data=buffer, file_name=f"industrial_noise_{sample_num + 1}.wav", mime="audio/wav")

            # Plot waveform
            st.markdown("#### 📈 Waveform")
            fig_waveform, ax = plt.subplots()
            times = np.linspace(0, duration, len(plot_data))
            if channels == "Stereo":
                ax.plot(times, plot_data[:,0], label='Left Channel', color='steelblue')
                ax.plot(times, plot_data[:,1], label='Right Channel', color='darkorange')
            else:
                ax.plot(times, plot_data.flatten(), color='steelblue')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            if channels == "Stereo":
                ax.legend()
            st.pyplot(fig_waveform)

            # Plot spectrum
            st.markdown("#### 📊 Frequency Spectrum")
            fig_spectrum, ax = plt.subplots()
            if channels == "Stereo":
                data_mono = plot_data.mean(axis=1)
            else:
                data_mono = plot_data.flatten()
            freqs = np.fft.rfftfreq(len(data_mono), 1/sample_rate)
            fft_magnitude = np.abs(np.fft.rfft(data_mono))
            ax.semilogx(freqs, fft_magnitude, color='darkorange')
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Magnitude")
            ax.grid(True)
            st.pyplot(fig_spectrum)

            # Plot spectrogram
            st.markdown("#### 🎼 Spectrogram")
            fig_spectrogram, ax = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(data_mono)), ref=np.max)
            img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
            ax.set_title('Spectrogram')
            fig_spectrogram.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig_spectrogram)

            st.markdown("---")  # Separator between samples

    else:
        st.write("Adjust parameters and click **Generate Noise** to create your industrial noise samples.")

# Run the app
if __name__ == "__main__":
    main()
