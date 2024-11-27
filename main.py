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
import tempfile
import zipfile

# For interactive plots
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Industrial Noise Generator Pro Max with Variations",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🎹 Industrial Noise Generator Pro Max with Variations")
st.markdown("""
Welcome to the **Industrial Noise Generator Pro Max with Variations**! This app allows you to generate multiple industrial noise samples with advanced features:

- **Signal Flow Customization:** Reroute effects and components in any order.
- **BPM Customization:** Set beats per minute for rhythm-based noise.
- **Note and Scale Selection:** Choose notes and scales for the synthesizer.
- **Built-in Synthesizer:** Create unique sounds with various waveforms.
- **Advanced Effects:** Apply reverb, delay, distortion, and more in any order.
- **Variations:** Automatically generate variations for each sample.
- **Interactive Waveform Editor:** Zoom, pan, and interact with your waveform.
- **Session Management:** Store and download all your generated sounds.
- **Audio Input Feature:** **Record or upload audio** to include in your samples creatively.

**Get started by adjusting the parameters below and click "Generate Noise" to create your samples!**
""")

# Initialize session state for temporary directory
if 'temp_dir' not in st.session_state:
    st.session_state['temp_dir'] = tempfile.TemporaryDirectory()
    st.session_state['generated_files'] = []

# Sidebar for preset selection and sample generation
with st.sidebar:
    st.header("🎚️ Presets")
    preset_options = ["Default", "Heavy Machinery", "Factory Floor", "Electric Hum", "Custom"]
    preset = st.selectbox(
        "Choose a Preset",
        preset_options,
        help="Select a preset to quickly load predefined settings."
    )

    # Number of samples to generate
    st.subheader("🔢 Sample Generation")
    num_samples = st.number_input(
        "Number of Samples to Generate",
        min_value=1,
        max_value=100,
        value=1,
        help="Select how many unique samples you want to generate."
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
                'waveform_frequencies': {},
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
                'waveform_frequencies': {"Square Wave": 55},
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
                'waveform_frequencies': {"Sawtooth Wave": 110},
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
                'waveform_frequencies': {"Sine Wave": 60},
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

# Main content area
st.markdown("---")

# Organize controls using tabs
tabs = st.tabs(["Basic Settings", "Advanced Settings", "Effects", "Synthesizer", "Session Management"])

# Basic Settings Tab
with tabs[0]:
    st.header("🎛️ Basic Settings")
    # Duration settings
    if preset_params:
        duration_type = preset_params.get('duration_type', 'Seconds')
    else:
        duration_type = 'Seconds'
    duration_type = st.selectbox(
        "Duration Type",
        ["Seconds", "Milliseconds", "Beats"],
        index=["Seconds", "Milliseconds", "Beats"].index(duration_type),
        help="Choose how you want to specify the duration."
    )
    if duration_type == "Seconds":
        duration = st.slider(
            "Duration (seconds)",
            min_value=1,
            max_value=60,
            value=preset_params['duration'] if preset_params else 5,
            help="Select the duration of the noise in seconds."
        )
    elif duration_type == "Milliseconds":
        duration_ms = st.slider(
            "Duration (milliseconds)",
            min_value=100,
            max_value=60000,
            value=int(preset_params['duration'] * 1000) if preset_params else 5000,
            help="Select the duration of the noise in milliseconds."
        )
        duration = duration_ms / 1000.0  # Convert to seconds
    elif duration_type == "Beats":
        bpm = st.slider(
            "BPM",
            min_value=30,
            max_value=300,
            value=preset_params['bpm'] if preset_params else 120,
            help="Set the beats per minute."
        )
        beats = st.slider(
            "Number of Beats",
            min_value=1,
            max_value=128,
            value=preset_params['beats'] if preset_params else 8,
            help="Set the number of beats."
        )
        duration = (60 / bpm) * beats  # Convert beats to seconds
    else:
        duration = 5

    noise_options = ["White Noise", "Pink Noise", "Brown Noise", "Blue Noise", "Violet Noise", "Grey Noise"]
    noise_types = st.multiselect(
        "Noise Types",
        noise_options,
        default=preset_params['noise_types'] if preset_params else ["White Noise"],
        help="Select one or more types of noise to include."
    )
    waveform_options = ["Sine Wave", "Square Wave", "Sawtooth Wave", "Triangle Wave"]
    waveform_types = st.multiselect(
        "Waveform Types",
        waveform_options,
        default=preset_params['waveform_types'] if preset_params else [],
        help="Select one or more waveforms to include."
    )

    # Collect waveform frequencies
    waveform_frequencies = {}
    for waveform_type in waveform_types:
        frequency = st.slider(
            f"{waveform_type} Frequency (Hz)",
            min_value=20,
            max_value=20000,
            value=preset_params['waveform_frequencies'].get(waveform_type, 440) if preset_params else 440,
            key=f"{waveform_type}_frequency_slider",
            help=f"Set the frequency for the {waveform_type.lower()}."
        )
        waveform_frequencies[waveform_type] = frequency

# Advanced Settings Tab
with tabs[1]:
    st.header("⚙️ Advanced Settings")
    lowcut = st.slider(
        "Low Cut Frequency (Hz)",
        min_value=20,
        max_value=10000,
        value=preset_params['lowcut'] if preset_params else 100,
        help="Set the low cut-off frequency for the bandpass filter."
    )
    highcut = st.slider(
        "High Cut Frequency (Hz)",
        min_value=1000,
        max_value=24000,
        value=preset_params['highcut'] if preset_params else 5000,
        help="Set the high cut-off frequency for the bandpass filter."
    )
    order = st.slider(
        "Filter Order",
        min_value=1,
        max_value=10,
        value=preset_params['order'] if preset_params else 6,
        help="Set the order of the bandpass filter."
    )
    amplitude = st.slider(
        "Amplitude",
        min_value=0.0,
        max_value=1.0,
        value=preset_params['amplitude'] if preset_params else 1.0,
        help="Set the amplitude (volume) of the generated noise."
    )
    sample_rate = st.selectbox(
        "Sample Rate (Hz)",
        [48000, 44100, 32000, 22050, 16000, 8000],
        index=[48000, 44100, 32000, 22050, 16000, 8000].index(preset_params['sample_rate']) if preset_params else 0,
        help="Choose the sample rate for the audio."
    )
    channels = st.selectbox(
        "Channels",
        ["Mono", "Stereo"],
        index=["Mono", "Stereo"].index(preset_params['channels']) if preset_params else 0,
        help="Select mono or stereo output."
    )
    fade_in = st.slider(
        "Fade In Duration (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=preset_params['fade_in'] if preset_params else 0.0,
        help="Set the duration for fade-in effect."
    )
    fade_out = st.slider(
        "Fade Out Duration (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=preset_params['fade_out'] if preset_params else 0.0,
        help="Set the duration for fade-out effect."
    )
    panning = st.slider(
        "Panning (Stereo Only)",
        min_value=0.0,
        max_value=1.0,
        value=preset_params['panning'] if preset_params else 0.5,
        help="Adjust the panning for stereo output."
    )
    bit_depth = st.selectbox(
        "Bit Depth",
        [16, 24, 32],
        index=[16, 24, 32].index(preset_params['bit_depth']) if preset_params else 0,
        help="Select the bit depth for the audio."
    )

# Effects Tab
with tabs[2]:
    st.header("🎚️ Effects Chain")
    modulation = st.selectbox(
        "Modulation",
        [None, "Amplitude Modulation", "Frequency Modulation"],
        index=[None, "Amplitude Modulation", "Frequency Modulation"].index(preset_params['modulation']) if preset_params else 0,
        help="Choose a modulation effect to apply."
    )
    reverse_audio = st.checkbox(
        "Enable Audio Reversal",
        value=preset_params['reverse_audio'] if preset_params else False,
        help="Reverse the audio for a unique effect."
    )
    bitcrusher = st.checkbox(
        "Enable Bitcrusher",
        value=preset_params['bitcrusher'] if preset_params else False,
        help="Apply a bitcrusher effect to reduce audio fidelity."
    )
    if bitcrusher:
        bitcrusher_depth = st.slider(
            "Bit Depth for Bitcrusher",
            min_value=1,
            max_value=16,
            value=preset_params['bitcrusher_depth'] if preset_params else 8,
            help="Set the bit depth for the bitcrusher effect."
        )
    else:
        bitcrusher_depth = 16
    sample_reduction = st.checkbox(
        "Enable Sample Rate Reduction",
        value=preset_params['sample_reduction'] if preset_params else False,
        help="Reduce the sample rate for a lo-fi effect."
    )
    if sample_reduction:
        sample_reduction_factor = st.slider(
            "Reduction Factor",
            min_value=1,
            max_value=16,
            value=preset_params['sample_reduction_factor'] if preset_params else 1,
            help="Set the factor by which to reduce the sample rate."
        )
    else:
        sample_reduction_factor = 1
    rhythmic_effect_options = ["Stutter", "Glitch"]
    rhythmic_effects = st.multiselect(
        "Select Rhythmic Effects",
        rhythmic_effect_options,
        default=preset_params['rhythmic_effects'] if preset_params else [],
        help="Choose rhythmic effects to apply."
    )
    st.subheader("Effects Chain")
    effect_options = ["None", "Reverb", "Delay", "Distortion", "Tremolo", "Chorus", "Flanger", "Phaser", "Compression", "EQ", "Pitch Shifter", "High-pass Filter", "Low-pass Filter", "Vibrato", "Auto-pan"]
    effects_chain = []
    for i in range(1, 6):
        effect = st.selectbox(
            f"Effect Slot {i}",
            effect_options,
            index=effect_options.index(preset_params['effects_chain'][i-1]) if preset_params and len(preset_params['effects_chain']) >= i else 0,
            key=f"effect_slot_{i}",
            help="Select an effect to apply at this position in the chain."
        )
        if effect != "None":
            effects_chain.append(effect)
    # Collect parameters for each effect
    effect_params = {}
    for effect in effects_chain:
        with st.expander(f"{effect} Parameters"):
            params = {}
            if effect == "Reverb":
                decay = st.slider(
                    "Reverb Decay",
                    0.1,
                    2.0,
                    preset_params['effect_params'][effect]['decay'] if preset_params and effect in preset_params['effect_params'] else 0.5,
                    key=f"reverb_decay_{effect}",
                    help="Set the decay time for the reverb effect."
                )
                params['decay'] = decay
            elif effect == "Delay":
                delay_time = st.slider(
                    "Delay Time (seconds)",
                    0.1,
                    1.0,
                    preset_params['effect_params'][effect]['delay_time'] if preset_params and effect in preset_params['effect_params'] else 0.5,
                    key=f"delay_time_{effect}",
                    help="Set the delay time for the delay effect."
                )
                feedback = st.slider(
                    "Delay Feedback",
                    0.0,
                    1.0,
                    preset_params['effect_params'][effect]['feedback'] if preset_params and effect in preset_params['effect_params'] else 0.5,
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
                    preset_params['effect_params'][effect]['gain'] if preset_params and effect in preset_params['effect_params'] else 20.0,
                    key=f"distortion_gain_{effect}",
                    help="Set the gain level for the distortion effect."
                )
                threshold = st.slider(
                    "Distortion Threshold",
                    0.0,
                    1.0,
                    preset_params['effect_params'][effect]['threshold'] if preset_params and effect in preset_params['effect_params'] else 0.5,
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
                    preset_params['effect_params'][effect]['rate'] if preset_params and effect in preset_params['effect_params'] else 5.0,
                    key=f"tremolo_rate_{effect}",
                    help="Set the rate for the tremolo effect."
                )
                depth = st.slider(
                    "Tremolo Depth",
                    0.0,
                    1.0,
                    preset_params['effect_params'][effect]['depth'] if preset_params and effect in preset_params['effect_params'] else 0.8,
                    key=f"tremolo_depth_{effect}",
                    help="Set the depth for the tremolo effect."
                )
                params['rate'] = rate
                params['depth'] = depth
            # Include other effect parameters as needed
            effect_params[effect] = params

# Synthesizer Tab
with tabs[3]:
    st.header("🎹 Synthesizer")
    synth_enabled = st.checkbox(
        "Enable Synthesizer",
        value=preset_params['synth_enabled'] if preset_params else False,
        help="Include synthesizer-generated tones."
    )
    if synth_enabled:
        note_options = ['C', 'C#', 'D', 'D#', 'E', 'F',
                        'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave_options = [str(i) for i in range(1, 8)]
        all_notes = [note + octave for octave in octave_options for note in note_options]
        selected_notes = st.multiselect(
            "Notes",
            all_notes,
            default=preset_params['synth_notes'] if preset_params else ['C4'],
            help="Select notes for the synthesizer."
        )
        synth_notes = selected_notes
        scale_options = ['Major', 'Minor', 'Pentatonic', 'Blues', 'Chromatic']
        synth_scale = st.selectbox(
            "Scale",
            scale_options,
            index=scale_options.index(preset_params['synth_scale']) if preset_params else 0,
            help="Choose a scale for the synthesizer."
        )
        synth_waveform = st.selectbox(
            "Waveform",
            ["Sine", "Square", "Sawtooth", "Triangle"],
            index=["Sine", "Square", "Sawtooth", "Triangle"].index(preset_params['synth_waveform']) if preset_params else 0,
            help="Select the waveform for the synthesizer."
        )
        st.markdown("**Envelope**")
        synth_attack = st.slider(
            "Attack",
            0.0,
            1.0,
            preset_params['synth_attack'] if preset_params else 0.01,
            help="Set the attack time for the envelope."
        )
        synth_decay = st.slider(
            "Decay",
            0.0,
            1.0,
            preset_params['synth_decay'] if preset_params else 0.1,
            help="Set the decay time for the envelope."
        )
        synth_sustain = st.slider(
            "Sustain",
            0.0,
            1.0,
            preset_params['synth_sustain'] if preset_params else 0.7,
            help="Set the sustain level for the envelope."
        )
        synth_release = st.slider(
            "Release",
            0.0,
            1.0,
            preset_params['synth_release'] if preset_params else 0.2,
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

    st.subheader("Arpeggiation and Sequencer")
    arpeggiation = st.checkbox(
        "Enable Arpeggiation",
        value=preset_params['arpeggiation'] if preset_params else False,
        help="Apply an arpeggiation effect to the notes."
    )
    sequencer = st.checkbox(
        "Enable Sequencer",
        value=preset_params['sequencer'] if preset_params else False,
        help="Sequence the generated tones."
    )
    if sequencer:
        sequence_patterns = ['Ascending', 'Descending', 'Random']
        sequence_pattern = st.selectbox(
            "Sequence Pattern",
            sequence_patterns,
            index=sequence_patterns.index(preset_params['sequence_pattern']) if preset_params else 0,
            help="Choose a pattern for the sequencer."
        )
        # Adjustable sequence duration
        sequence_duration = st.slider(
            "Sequence Duration (seconds)",
            min_value=0.1,
            max_value=duration,
            value=0.5,
            help="Set the duration for each sequence in the sequencer."
        )
    else:
        sequence_pattern = 'Random'
        sequence_duration = 0.5

# Session Management Tab
with tabs[4]:
    st.header("💾 Session Management")
    st.subheader("Save Current Preset")
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
            'waveform_frequencies': waveform_frequencies,
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
            'effects_chain': effects_chain,
            'effect_params': effect_params,
            'voice_changer': False,
            'pitch_shift_semitones': 0,
            'algorithmic_composition': False,
            'composition_type': None,
            'synth_enabled': synth_enabled,
            'synth_notes': synth_notes,
            'synth_scale': synth_scale,
            'synth_waveform': synth_waveform,
            'synth_attack': synth_attack,
            'synth_decay': synth_decay,
            'synth_sustain': synth_sustain,
            'synth_release': synth_release
        }
        if not os.path.exists('presets'):
            os.makedirs('presets')
        with open(f'presets/{preset_name}.pkl', 'wb') as f:
            pickle.dump(current_params, f)
        st.success(f"Preset '{preset_name}' saved!")

    st.subheader("Load Preset")
    def list_presets():
        if os.path.exists('presets'):
            return [f.replace('.pkl', '') for f in os.listdir('presets') if f.endswith('.pkl')]
        else:
            return []

    def load_preset(name):
        with open(f'presets/{name}.pkl', 'rb') as f:
            params = pickle.load(f)
        return params

    available_presets = list_presets()
    if available_presets:
        load_preset_name = st.selectbox(
            "Select Preset",
            available_presets,
            help="Select a preset to load."
        )
        if st.button("Load Selected Preset"):
            loaded_params = load_preset(load_preset_name)
            st.session_state['loaded_params'] = loaded_params
            st.success(f"Preset '{load_preset_name}' loaded!")
            st.experimental_rerun()
    else:
        st.info("No presets available.")

    # **Audio Input Feature**
    st.header("🎤 Audio Input")
    st.markdown("""
    **Record or upload audio** to creatively include it in your noise samples. This feature allows you to:
    - Incorporate your own sounds into the generated noise.
    - Apply effects and transformations to your audio input.
    - Experiment with layering and mixing for unique results.
    """)
    audio_data = st.file_uploader("Record or Upload Audio", type=["wav", "mp3"])

    if audio_data:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_data.read())
            audio_path = temp_audio_file.name

        # Process audio (e.g., display, transform)
        st.audio(audio_path)
        st.write("Audio saved to:", audio_path)

        # Optionally include the uploaded audio in the noise generation
        include_audio = st.checkbox(
            "Include this audio in the generated samples",
            value=True,
            help="When selected, the uploaded audio will be mixed into the generated noise samples."
        )
    else:
        audio_path = None
        include_audio = False

# Generate Noise Button
st.markdown("---")
if st.button("🎶 Generate Noise"):
    # Functions to generate noise
    def generate_white_noise(duration, sample_rate):
        samples = np.random.normal(0, 1, int(duration * sample_rate))
        return samples

    def generate_pink_noise(duration, sample_rate):
        # Voss-McCartney algorithm
        samples = int(duration * sample_rate)
        n_rows = 16
        n_columns = int(np.ceil(samples / n_rows))
        array = np.random.randn(n_rows, n_columns)
        cumulative = np.cumsum(array, axis=0)
        pink_noise = cumulative[-1, :]
        pink_noise = pink_noise[:samples]
        return pink_noise

    def generate_brown_noise(duration, sample_rate):
        samples = int(duration * sample_rate)
        brown_noise = np.cumsum(np.random.randn(samples))
        brown_noise = brown_noise / np.max(np.abs(brown_noise) + 1e-7)
        return brown_noise

    def generate_blue_noise(duration, sample_rate):
        # Differentiated white noise
        samples = int(duration * sample_rate)
        white = np.random.normal(0, 1, samples)
        blue_noise = np.diff(white)
        blue_noise = np.concatenate(([0], blue_noise))
        return blue_noise

    def generate_violet_noise(duration, sample_rate):
        # Differentiated blue noise
        samples = int(duration * sample_rate)
        white = np.random.normal(0, 1, samples)
        violet_noise = np.diff(np.diff(white))
        violet_noise = np.concatenate(([0, 0], violet_noise))
        return violet_noise

    def generate_grey_noise(duration, sample_rate):
        # Shaped white noise to match human hearing
        samples = int(duration * sample_rate)
        white = np.random.normal(0, 1, samples)
        freqs = np.fft.rfftfreq(samples, 1/sample_rate)
        a_weighting = (12200**2 * freqs**4) / ((freqs**2 + 20.6**2) * np.sqrt((freqs**2 + 107.7**2) * (freqs**2 + 737.9**2)) * (freqs**2 + 12200**2))
        a_weighting = a_weighting / np.max(a_weighting + 1e-7)
        white_fft = np.fft.rfft(white)
        grey_fft = white_fft * a_weighting
        grey_noise = np.fft.irfft(grey_fft)
        return grey_noise

    # Functions to generate waveforms
    def generate_waveform(waveform_type, frequency, duration, sample_rate):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        if waveform_type == "Sine Wave":
            waveform = np.sin(2 * np.pi * frequency * t)
        elif waveform_type == "Square Wave":
            waveform = signal.square(2 * np.pi * frequency * t)
        elif waveform_type == "Sawtooth Wave":
            waveform = signal.sawtooth(2 * np.pi * frequency * t)
        elif waveform_type == "Triangle Wave":
            waveform = signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
        else:
            waveform = np.zeros_like(t)
        return waveform

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
        if envelope is not None:
            tone *= envelope
        return tone

    def note_to_freq(note):
        # Convert note (e.g., 'A4') to frequency
        A4_freq = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(note[-1])
        key_number = note_names.index(note[:-1])
        n = key_number + (octave - 4) * 12
        freq = A4_freq * (2 ** (n / 12))
        return freq

    def generate_envelope(total_samples, sample_rate, attack, decay, sustain, release):
        envelope = np.zeros(total_samples)
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        release_samples = int(release * sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        # Ensure no negative values
        sustain_samples = max(sustain_samples, 0)
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
        # Sustain
        envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain
        # Release
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
        return envelope

    def butter_bandpass(lowcut, highcut, fs, order=5):
        # Create a bandpass filter
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(data, lowcut, highcut, fs, order=5):
        # Apply the bandpass filter
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def apply_fade(data, sample_rate, fade_in, fade_out):
        # Ensure data is at least two-dimensional
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Apply fade in/out
        total_samples = data.shape[0]
        fade_in_samples = int(fade_in * sample_rate)
        fade_out_samples = int(fade_out * sample_rate)

        # Ensure fade samples do not exceed total samples
        fade_in_samples = min(fade_in_samples, total_samples)
        fade_out_samples = min(fade_out_samples, total_samples)

        # Apply fade-in if applicable
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples).reshape(-1, 1)
            data[:fade_in_samples] *= fade_in_curve

        # Apply fade-out if applicable
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples).reshape(-1, 1)
            data[-fade_out_samples:] *= fade_out_curve

        # Flatten data if original was one-dimensional
        if data.shape[1] == 1:
            data = data.flatten()

        return data

    def apply_amplitude_modulation(data, sample_rate, mod_freq=5.0):
        # Apply amplitude modulation
        t = np.linspace(0, len(data)/sample_rate, num=len(data))
        modulator = np.sin(2 * np.pi * mod_freq * t)
        return data * modulator

    def apply_frequency_modulation(data, sample_rate, mod_freq=5.0, mod_index=2.0):
        # Apply frequency modulation
        t = np.linspace(0, len(data)/sample_rate, num=len(data))
        carrier = np.sin(2 * np.pi * 440 * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
        return data * carrier

    def apply_reverb(data, sample_rate, decay=0.5):
        # Simple reverb effect using feedback delay
        reverb_data = np.copy(data)
        delay_samples = int(0.02 * sample_rate)
        for i in range(delay_samples, len(data)):
            reverb_data[i] += decay * reverb_data[i - delay_samples]
        return reverb_data

    def apply_delay(data, sample_rate, delay_time=0.5, feedback=0.5):
        # Simple delay effect
        delay_samples = int(delay_time * sample_rate)
        delayed_data = np.zeros(len(data) + delay_samples)
        delayed_data[:len(data)] = data
        for i in range(len(data)):
            delayed_data[i + delay_samples] += data[i] * feedback
        return delayed_data[:len(data)]

    def apply_distortion(data, gain=20, threshold=0.5):
        # Simple distortion effect
        data = data * gain
        data[data > threshold] = threshold
        data[data < -threshold] = -threshold
        return data

    def adjust_bit_depth(data, bit_depth):
        # Adjust bit depth
        max_val = 2 ** (bit_depth - 1) - 1
        data = data * max_val
        data = np.round(data)
        data = data / max_val
        return data

    def pan_stereo(data, panning):
        # Panning for stereo sound
        left = data * (1 - panning)
        right = data * panning
        stereo_data = np.vstack((left, right)).T
        return stereo_data

    def generate_algorithmic_composition(duration, sample_rate, composition_type):
        if composition_type == "Random Melody":
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            melody = np.zeros_like(t)
            num_notes = int(duration * 2)  # 2 notes per second
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
                end_sample = start_sample + int(0.1 * sample_rate)  # 100 ms per beat
                if end_sample > len(t):
                    end_sample = len(t)
                rhythm[start_sample:end_sample] = 1.0
            return rhythm * np.sin(2 * np.pi * 440 * t)
        else:
            return np.zeros(int(duration * sample_rate))

    def pitch_shift_audio(data, sample_rate, n_steps):
        # Pitch shift using librosa
        return librosa.effects.pitch_shift(data, sample_rate, n_steps=n_steps)

    def apply_tremolo(data, sample_rate, rate=5.0, depth=0.8):
        # Tremolo effect
        t = np.arange(len(data)) / sample_rate
        tremolo = (1 + depth * np.sin(2 * np.pi * rate * t)) / 2
        return data * tremolo

    def apply_bitcrusher(data, bit_depth):
        # Bitcrusher effect
        max_val = 2 ** (bit_depth - 1) - 1
        data = data * max_val
        data = np.round(data)
        data = data / max_val
        return data

    def apply_sample_reduction(data, reduction_factor):
        # Sample rate reduction
        reduced_data = data[::reduction_factor]
        upsampled_data = np.repeat(reduced_data, reduction_factor)
        upsampled_data = upsampled_data[:len(data)]  # Ensure the length matches
        return upsampled_data

    def apply_reverse(data):
        # Reverse audio
        return data[::-1]

    def apply_stutter(data, sample_rate, interval=0.1):
        # Stutter effect
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
        # Glitch effect
        glitch_length = int(0.05 * sample_rate)
        glitch_data = np.copy(data)
        for i in range(0, len(data), glitch_length * 4):
            glitch_data[i:i+glitch_length] = 0
        return glitch_data

    def apply_arpeggiation(data, sample_rate, pattern='Ascending'):
        # Arpeggiation effect
        num_notes = 4
        note_length = int(len(data) / num_notes)
        arpeggiated_data = np.zeros_like(data)
        indices = list(range(num_notes))
        if pattern == 'Ascending':
            pass  # Indices already in ascending order
        elif pattern == 'Descending':
            indices = indices[::-1]
        elif pattern == 'Random':
            random.shuffle(indices)
        for idx, i in enumerate(indices):
            start = i * note_length
            end = start + note_length
            arpeggiated_data[start:end] = data[start:end]
        return arpeggiated_data

    def apply_sequencer(data, sample_rate, pattern='Random', sequence_duration=0.5):
        # Sequencer effect with adjustable sequence duration
        sequence_length = int(sample_rate * sequence_duration)
        if sequence_length > len(data):
            sequence_length = len(data)
        num_sequences = max(len(data) // sequence_length, 1)
        sequences = [data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]
        # Handle any remaining samples
        remaining_samples = len(data) % sequence_length
        if remaining_samples > 0 and len(sequences) > 0:
            sequences.append(data[-remaining_samples:])
        if pattern == 'Ascending':
            pass
        elif pattern == 'Descending':
            sequences = sequences[::-1]
        elif pattern == 'Random':
            random.shuffle(sequences)
        sequenced_data = np.concatenate(sequences)
        return sequenced_data

    def apply_chorus(data, sample_rate, rate=1.5, depth=0.5):
        # Simple chorus effect
        delay_samples = int((1 / rate) * sample_rate)
        modulated_delay = depth * np.sin(2 * np.pi * rate * np.arange(len(data)) / sample_rate)
        delayed_data = np.zeros_like(data)
        for i in range(len(data)):
            delay = int(delay_samples * (1 + modulated_delay[i]))
            if i - delay >= 0:
                delayed_data[i] = data[i - delay]
        return data + delayed_data

    def apply_flanger(data, sample_rate, rate=0.5, depth=0.7):
        # Simple flanger effect
        delay_samples = int(0.001 * sample_rate)  # 1 ms delay
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
        # Simple phaser effect
        phaser_data = np.copy(data)
        phase = depth * np.sin(2 * np.pi * rate * np.arange(len(data)) / sample_rate)
        phaser_data = np.sin(2 * np.pi * phaser_data + phase)
        return phaser_data

    def apply_compression(data, threshold=0.5, ratio=2.0):
        # Simple compression effect
        compressed_data = np.copy(data)
        over_threshold = np.abs(compressed_data) > threshold
        compressed_data[over_threshold] = threshold + (compressed_data[over_threshold] - threshold) / ratio
        compressed_data[compressed_data < -threshold] = -threshold + (compressed_data[compressed_data < -threshold] + threshold) / ratio
        return compressed_data

    def apply_eq(data, sample_rate, bands):
        # Simple 3-band EQ
        eq_data = np.copy(data)
        # Low frequencies (20 Hz - 250 Hz)
        b, a = butter(2, [20 / (0.5 * sample_rate), 250 / (0.5 * sample_rate)], btype='band')
        low = lfilter(b, a, data) * (10 ** (bands['low'] / 20))
        # Mid frequencies (250 Hz - 4 kHz)
        b, a = butter(2, [250 / (0.5 * sample_rate), 4000 / (0.5 * sample_rate)], btype='band')
        mid = lfilter(b, a, data) * (10 ** (bands['mid'] / 20))
        # High frequencies (4 kHz - 20 kHz)
        b, a = butter(2, [4000 / (0.5 * sample_rate), 20000 / (0.5 * sample_rate)], btype='band')
        high = lfilter(b, a, data) * (10 ** (bands['high'] / 20))
        eq_data = low + mid + high
        return eq_data

    def apply_pitch_shift(data, sample_rate, semitones):
        # Pitch shifting using librosa
        return librosa.effects.pitch_shift(data, sample_rate, n_steps=semitones)

    def apply_highpass_filter(data, sample_rate, cutoff):
        # High-pass filter
        b, a = butter(2, cutoff / (0.5 * sample_rate), btype='high')
        return lfilter(b, a, data)

    def apply_lowpass_filter(data, sample_rate, cutoff):
        # Low-pass filter
        b, a = butter(2, cutoff / (0.5 * sample_rate), btype='low')
        return lfilter(b, a, data)

    def apply_vibrato(data, sample_rate, rate=5.0, depth=0.5):
        # Vibrato effect
        t = np.arange(len(data))
        vibrato = np.sin(2 * np.pi * rate * t / sample_rate)
        indices = t + depth * vibrato * sample_rate
        indices = np.clip(indices, 0, len(data) - 1).astype(int)
        return data[indices]

    def apply_autopan(data, sample_rate, rate=1.0):
        # Auto-pan effect
        t = np.arange(len(data)) / sample_rate
        pan = 0.5 * (1 + np.sin(2 * np.pi * rate * t))
        left = data * pan
        right = data * (1 - pan)
        stereo_data = np.vstack((left, right)).T
        return stereo_data

    # Main noise generation logic
    progress_bar = st.progress(0)
    status_text = st.empty()
    for sample_num in range(num_samples):
        status_text.text(f"Generating sample {sample_num + 1} of {num_samples}...")
        st.markdown(f"### Sample {sample_num + 1}")
        # Generate noise based on selection
        total_samples = int(duration * sample_rate)
        combined_data = np.zeros(total_samples, dtype=np.float32)  # Ensure float type

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
            data = data.astype(np.float32)  # Ensure float type

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
            # Apply variation to frequency
            frequency *= frequency_variation
            data = generate_waveform(waveform_type, frequency, duration, sample_rate)
            # Apply filter with varied parameters
            data = apply_filter(data, varied_lowcut, varied_highcut, sample_rate, order)
            # Normalize audio
            data = data / np.max(np.abs(data) + 1e-7)
            data = data.astype(np.float32)  # Ensure float type

            # Adjust lengths if necessary
            if len(data) != len(combined_data):
                min_length = min(len(data), len(combined_data))
                data = data[:min_length]
                combined_data = combined_data[:min_length]

            # Combine waveforms
            combined_data += data

        # Synthesizer
        if synth_enabled:
            envelope = generate_envelope(total_samples, sample_rate, synth_attack, synth_decay, synth_sustain, synth_release)
            synth_data = np.zeros(total_samples, dtype=np.float32)
            for note in synth_notes:
                # Apply variation to note frequency
                note_freq = note_to_freq(note) * frequency_variation
                # Generate tone with varied frequency
                tone = generate_waveform(synth_waveform + " Wave", note_freq, duration, sample_rate)
                # Apply envelope
                tone *= envelope
                synth_data += tone
            synth_data = synth_data / np.max(np.abs(synth_data) + 1e-7)
            synth_data = synth_data.astype(np.float32)  # Ensure float type

            # Adjust lengths if necessary
            if len(synth_data) != len(combined_data):
                min_length = min(len(synth_data), len(combined_data))
                synth_data = synth_data[:min_length]
                combined_data = combined_data[:min_length]

            combined_data += synth_data

        # Include uploaded audio file
        if audio_path and include_audio:
            y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
            y = y[:total_samples]  # Ensure length matches
            y = y / np.max(np.abs(y) + 1e-7)  # Normalize
            y = y.astype(np.float32)  # Ensure float type

            # Adjust lengths if necessary
            if len(y) != len(combined_data):
                min_length = min(len(y), len(combined_data))
                y = y[:min_length]
                combined_data = combined_data[:min_length]

            combined_data += y

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
            combined_data = apply_sequencer(combined_data, sample_rate, pattern=sequence_pattern, sequence_duration=sequence_duration)
            plot_data = apply_sequencer(plot_data, sample_rate, pattern=sequence_pattern, sequence_duration=sequence_duration)

        # Apply effects in chain order
        for effect in effects_chain:
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
            # Include other effects as needed

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
        D = librosa.stft(plot_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Spectrogram')
        st.pyplot(fig)

        # Update progress
        progress_bar.progress((sample_num + 1) / num_samples)

    status_text.text("All samples generated!")

    # Download all generated files as a zip
    if st.session_state['generated_files']:
        zip_filename = "industrial_noise_samples.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in st.session_state['generated_files']:
                zipf.write(file, os.path.basename(file))

        with open(zip_filename, "rb") as f:
            btn = st.download_button(
                label="Download All Samples",
                data=f.read(),
                file_name=zip_filename,
                mime="application/zip"
            )
