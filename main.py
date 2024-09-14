import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
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
num_samples = st.sidebar.number_input("Number of Samples to Generate", min_value=1, max_value=100, value=1)

# Presets
preset_options = ["Default", "Heavy Machinery", "Factory Floor", "Electric Hum", "Custom"]
preset = st.sidebar.selectbox("Choose a Preset", preset_options)

# Function to set preset parameters
def set_preset(preset):
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
        'synth_enabled': False,
        'synth_notes': ['C4'],
        'synth_scale': 'Major',
        'synth_waveform': 'Sine',
        'synth_attack': 0.01,
        'synth_decay': 0.1,
        'synth_sustain': 0.7,
        'synth_release': 0.2
    }
    
    if preset == "Heavy Machinery":
        params.update({
            'duration': 10,
            'noise_types': ["Brown Noise", "Pink Noise"],
            'lowcut': 50,
            'highcut': 2000,
            'order': 8,
            'channels': "Stereo",
            'fade_in': 1.0,
            'fade_out': 1.0,
            'effects': ["Reverb", "Distortion"],
            'effect_params': {'Reverb': {'decay': 0.8}, 'Distortion': {'gain': 15, 'threshold': 0.3}},
            'modulation': "Amplitude Modulation"
        })
    elif preset == "Factory Floor":
        params.update({
            'duration': 8,
            'noise_types': ["White Noise", "Brown Noise"],
            'lowcut': 150,
            'highcut': 4000,
            'order': 4,
            'amplitude': 0.8,
            'fade_in': 0.5,
            'fade_out': 0.5,
            'effects': ["Reverb"],
            'effect_params': {'Reverb': {'decay': 0.6}},
            'rhythmic_effects': ["Stutter"]
        })
    elif preset == "Electric Hum":
        params.update({
            'duration': 5,
            'noise_types': [],
            'waveform_types': ["Sine Wave"],
            'lowcut': 5000,
            'highcut': 20000,
            'order': 6,
            'amplitude': 0.6,
            'fade_in': 0.2,
            'fade_out': 0.2,
            'modulation': "Frequency Modulation"
        })
    
    return params

# Get parameters based on preset or custom settings
params = set_preset(preset)

if preset == "Custom":
    # Custom parameters
    duration_type = st.sidebar.selectbox("Duration Type", ["Seconds", "Milliseconds", "Beats"])
    if duration_type == "Seconds":
        params['duration'] = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=5)
    elif duration_type == "Milliseconds":
        duration_ms = st.sidebar.slider("Duration (milliseconds)", min_value=100, max_value=60000, value=5000)
        params['duration'] = duration_ms / 1000.0  # Convert to seconds
    elif duration_type == "Beats":
        params['bpm'] = st.sidebar.slider("BPM", min_value=30, max_value=300, value=120)
        params['beats'] = st.sidebar.slider("Number of Beats", min_value=1, max_value=128, value=8)
        params['duration'] = (60 / params['bpm']) * params['beats']  # Convert beats to seconds

    noise_options = ["White Noise", "Pink Noise", "Brown Noise", "Blue Noise", "Violet Noise", "Grey Noise"]
    params['noise_types'] = st.sidebar.multiselect("Noise Types", noise_options)
    waveform_options = ["Sine Wave", "Square Wave", "Sawtooth Wave", "Triangle Wave"]
    params['waveform_types'] = st.sidebar.multiselect("Waveform Types", waveform_options)
    
    # Single frequency slider for all waveforms
    if params['waveform_types']:
        params['waveform_frequency'] = st.sidebar.slider("Waveform Frequency (Hz)", min_value=20, max_value=20000, value=440)
    
    params['lowcut'] = st.sidebar.slider("Low Cut Frequency (Hz)", min_value=20, max_value=10000, value=100)
    params['highcut'] = st.sidebar.slider("High Cut Frequency (Hz)", min_value=1000, max_value=24000, value=5000)
    params['order'] = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=6)
    params['amplitude'] = st.sidebar.slider("Amplitude", min_value=0.0, max_value=1.0, value=1.0)
    params['sample_rate'] = st.sidebar.selectbox("Sample Rate (Hz)", [48000, 44100, 32000, 22050, 16000, 8000], index=0)
    params['channels'] = st.sidebar.selectbox("Channels", ["Mono", "Stereo"])
    params['fade_in'] = st.sidebar.slider("Fade In Duration (seconds)", min_value=0.0, max_value=5.0, value=0.0)
    params['fade_out'] = st.sidebar.slider("Fade Out Duration (seconds)", min_value=0.0, max_value=5.0, value=0.0)
    params['panning'] = st.sidebar.slider("Panning (Stereo Only)", min_value=0.0, max_value=1.0, value=0.5)
    params['bit_depth'] = st.sidebar.selectbox("Bit Depth", [16, 24, 32], index=0)
    params['modulation'] = st.sidebar.selectbox("Modulation", [None, "Amplitude Modulation", "Frequency Modulation"])
    params['reverse_audio'] = st.sidebar.checkbox("Enable Audio Reversal")
    params['bitcrusher'] = st.sidebar.checkbox("Enable Bitcrusher")
    if params['bitcrusher']:
        params['bitcrusher_depth'] = st.sidebar.slider("Bit Depth for Bitcrusher", min_value=1, max_value=16, value=8)
    params['sample_reduction'] = st.sidebar.checkbox("Enable Sample Rate Reduction")
    if params['sample_reduction']:
        params['sample_reduction_factor'] = st.sidebar.slider("Reduction Factor", min_value=1, max_value=16, value=1)
    rhythmic_effect_options = ["Stutter", "Glitch"]
    params['rhythmic_effects'] = st.sidebar.multiselect("Select Rhythmic Effects", rhythmic_effect_options)
    params['synth_enabled'] = st.sidebar.checkbox("Enable Synthesizer")
    if params['synth_enabled']:
        note_options = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave_options = [str(i) for i in range(1, 8)]
        params['synth_notes'] = st.sidebar.multiselect("Notes", [note + octave for octave in octave_options for note in note_options], default=['C4'])
        params['synth_scale'] = st.sidebar.selectbox("Scale", ['Major', 'Minor', 'Pentatonic', 'Blues', 'Chromatic'])
        params['synth_waveform'] = st.sidebar.selectbox("Waveform", ["Sine", "Square", "Sawtooth", "Triangle"])
        st.sidebar.markdown("**Envelope**")
        params['synth_attack'] = st.sidebar.slider("Attack", 0.0, 1.0, 0.01)
        params['synth_decay'] = st.sidebar.slider("Decay", 0.0, 1.0, 0.1)
        params['synth_sustain'] = st.sidebar.slider("Sustain", 0.0, 1.0, 0.7)
        params['synth_release'] = st.sidebar.slider("Release", 0.0, 1.0, 0.2)
    params['arpeggiation'] = st.sidebar.checkbox("Enable Arpeggiation")
    params['sequencer'] = st.sidebar.checkbox("Enable Sequencer")
    if params['sequencer']:
        params['sequence_pattern'] = st.sidebar.selectbox("Sequence Pattern", ['Ascending', 'Descending', 'Random'])
    effect_options = ["Reverb", "Delay", "Distortion", "Tremolo"]
    params['effects'] = st.sidebar.multiselect("Select Effects", effect_options)
    params['effect_params'] = {}
    for effect in params['effects']:
        st.sidebar.markdown(f"**{effect} Parameters**")
        if effect == "Reverb":
            params['effect_params']['Reverb'] = {'decay': st.sidebar.slider("Reverb Decay", 0.1, 2.0, 0.5)}
        elif effect == "Delay":
            params['effect_params']['Delay'] = {
                'delay_time': st.sidebar.slider("Delay Time (seconds)", 0.1, 1.0, 0.5),
                'feedback': st.sidebar.slider("Delay Feedback", 0.0, 1.0, 0.5)
            }
        elif effect == "Distortion":
            params['effect_params']['Distortion'] = {
                'gain': st.sidebar.slider("Distortion Gain", 1.0, 50.0, 20.0),
                'threshold': st.sidebar.slider("Distortion Threshold", 0.0, 1.0, 0.5)
            }
        elif effect == "Tremolo":
            params['effect_params']['Tremolo'] = {
                'rate': st.sidebar.slider("Tremolo Rate (Hz)", 0.1, 20.0, 5.0),
                'depth': st.sidebar.slider("Tremolo Depth", 0.0, 1.0, 0.8)
            }

# Functions to generate noise and apply effects
def generate_noise(noise_type, duration, sample_rate):
    samples = int(duration * sample_rate)
    if noise_type == "White Noise":
        return np.random.normal(0, 1, samples)
    elif noise_type == "Pink Noise":
        X = np.fft.rfft(np.random.normal(0, 1, samples))
        S = np.sqrt(np.arange(X.size)+1.)
        y = np.fft.irfft(X/S)
        return y
    elif noise_type == "Brown Noise":
        X = np.fft.rfft(np.random.normal(0, 1, samples))
        S = np.arange(X.size)+1
        y = np.fft.irfft(X/S)
        return y
    elif noise_type == "Blue Noise":
        X = np.fft.rfft(np.random.normal(0, 1, samples))
        S = np.sqrt(np.arange(X.size))
        y = np.fft.irfft(X*S)
        return y
    elif noise_type == "Violet Noise":
        X = np.fft.rfft(np.random.normal(0, 1, samples))
        S = np.arange(X.size)
        y = np.fft.irfft(X*S)
        return y
    elif noise_type == "Grey Noise":
        X = np.fft.rfft(np.random.normal(0, 1, samples))
        A = np.ones(X.size)
        A[1:] = np.sqrt(np.log(np.arange(1, X.size) + 1))
        y = np.fft.irfft(X*A)
        return y
    else:
        return np.zeros(samples)

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

def apply_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def apply_fade(data, sample_rate, fade_in, fade_out):
    total_samples = len(data)
    fade_in_samples = int(fade_in * sample_rate)
    fade_out_samples = int(fade_out * sample_rate)
    fade_in_curve = np.linspace(0, 1, fade_in_samples)
    fade_out_curve = np.linspace(1, 0, fade_out_samples)
    data[:fade_in_samples] *= fade_in_curve
    data[-fade_out_samples:] *= fade_out_curve
    return data

def apply_modulation(data, sample_rate, mod_type, mod_freq=5.0, mod_index=2.0):
    t = np.linspace(0, len(data)/sample_rate, num=len(data))
    if mod_type == "Amplitude Modulation":
        modulator = np.sin(2 * np.pi * mod_freq * t)
        return data * modulator
    elif mod_type == "Frequency Modulation":
        carrier = np.sin(2 * np.pi * 440 * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
        return data * carrier
    else:
        return data

def apply_effects(data, sample_rate, effects, effect_params):
    for effect in effects:
        if effect == "Reverb":
            data = apply_reverb(data, sample_rate, effect_params['Reverb']['decay'])
        elif effect == "Delay":
            data = apply_delay(data, sample_rate, effect_params['Delay']['delay_time'], effect_params['Delay']['feedback'])
        elif effect == "Distortion":
            data = apply_distortion(data, effect_params['Distortion']['gain'], effect_params['Distortion']['threshold'])
        elif effect == "Tremolo":
            data = apply_tremolo(data, sample_rate, effect_params['Tremolo']['rate'], effect_params['Tremolo']['depth'])
    return data

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

def apply_tremolo(data, sample_rate, rate=5.0, depth=0.8):
    t = np.arange(len(data)) / sample_rate
    tremolo = (1 + depth * np.sin(2 * np.pi * rate * t)) / 2
    return data * tremolo

def apply_bitcrusher(data, bit_depth):
    max_val = 2 ** (bit_depth - 1) - 1
    data = np.round(data * max_val) / max_val
    return data

def apply_sample_reduction(data, reduction_factor):
    return signal.resample(data, len(data) // reduction_factor)

def apply_rhythmic_effects(data, sample_rate, effects):
    for effect in effects:
        if effect == "Stutter":
            data = apply_stutter(data, sample_rate)
        elif effect == "Glitch":
            data = apply_glitch(data, sample_rate)
    return data

def apply_stutter(data, sample_rate, interval=0.1):
    stutter_samples = int(interval * sample_rate)
    num_repeats = 3
    stuttered_data = []
    for i in range(0, len(data), stutter_samples):
        chunk = data[i:i+stutter_samples]
        stuttered_data.extend([chunk] * num_repeats)
    return np.concatenate(stuttered_data)[:len(data)]

def apply_glitch(data, sample_rate):
    glitch_length = int(0.05 * sample_rate)
    glitch_data = np.copy(data)
    for i in range(0, len(data), glitch_length * 4):
        glitch_data[i:i+glitch_length] = 0
    return glitch_data

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
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    note_name = note[:-1]
    semitones = note_names.index(note_name)
    return 440 * (2 ** ((semitones - 9) / 12)) * (2 ** (octave - 4))

def generate_envelope(total_samples, attack, decay, sustain, release):
    envelope = np.zeros(total_samples)
    attack_samples = int(attack * total_samples)
    decay_samples = int(decay * total_samples)
    release_samples = int(release * total_samples)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
    envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain
    envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
    return envelope

def main():
    if st.button("🎶 Generate Noise"):
        for sample_num in range(num_samples):
            st.markdown(f"### Sample {sample_num + 1}")
            
            # Generate noise based on selection
            total_samples = int(params['duration'] * params['sample_rate'])
            combined_data = np.zeros(total_samples)

            # Generate noise types
            for noise_type in params['noise_types']:
                data = generate_noise(noise_type, params['duration'], params['sample_rate'])
                data = apply_filter(data, params['lowcut'], params['highcut'], params['sample_rate'], params['order'])
                combined_data += data

            # Generate waveform types
            for waveform_type in params['waveform_types']:
                data = generate_waveform(waveform_type, params['waveform_frequency'], params['duration'], params['sample_rate'])
                data = apply_filter(data, params['lowcut'], params['highcut'], params['sample_rate'], params['order'])
                combined_data += data

            # Synthesizer
            if params['synth_enabled']:
                envelope = generate_envelope(total_samples, params['synth_attack'], params['synth_decay'], params['synth_sustain'], params['synth_release'])
                for note in params['synth_notes']:
                    tone = generate_synth_tone(note, params['duration'], params['sample_rate'], params['synth_waveform'], envelope)
                    combined_data += tone

            # Normalize combined data
            combined_data = combined_data / np.max(np.abs(combined_data))

            # Apply amplitude
            combined_data *= params['amplitude']

            # Apply modulation
            if params['modulation']:
                combined_data = apply_modulation(combined_data, params['sample_rate'], params['modulation'])

            # Apply fade in/out
            combined_data = apply_fade(combined_data, params['sample_rate'], params['fade_in'], params['fade_out'])

            # Apply effects
            combined_data = apply_effects(combined_data, params['sample_rate'], params['effects'], params['effect_params'])

            # Apply rhythmic effects
            combined_data = apply_rhythmic_effects(combined_data, params['sample_rate'], params['rhythmic_effects'])

            # Apply bitcrusher
            if params['bitcrusher']:
                combined_data = apply_bitcrusher(combined_data, params['bitcrusher_depth'])

            # Apply sample rate reduction
            if params['sample_reduction']:
                combined_data = apply_sample_reduction(combined_data, params['sample_reduction_factor'])

            # Handle stereo or mono
            if params['channels'] == "Stereo":
                combined_data = np.column_stack((combined_data * (1 - params['panning']), combined_data * params['panning']))
            else:
                combined_data = combined_data.reshape(-1, 1)

            # Convert to proper dtype for saving
            if params['bit_depth'] == 16:
                combined_data = (combined_data * 32767).astype(np.int16)
            elif params['bit_depth'] == 24:
                combined_data = (combined_data * 8388607).astype(np.int32)
            else:  # 32-bit
                combined_data = combined_data.astype(np.float32)

            # Save audio to buffer
            buffer = io.BytesIO()
            if params['bit_depth'] == 24:
                sf.write(buffer, combined_data, params['sample_rate'], subtype='PCM_24')
            else:
                write(buffer, params['sample_rate'], combined_data)
            buffer.seek(0)

            # Play audio
            st.audio(buffer, format='audio/wav')

            # Provide download button
            st.download_button(label=f"💾 Download Sample {sample_num + 1}", data=buffer, file_name=f"industrial_noise_{sample_num + 1}.wav", mime="audio/wav")

            # Plot waveform
            st.markdown("#### 📈 Waveform")
            fig_waveform, ax = plt.subplots()
            times = np.linspace(0, params['duration'], len(combined_data))
            if params['channels'] == "Stereo":
                ax.plot(times, combined_data[:,0], label='Left Channel', color='steelblue')
                ax.plot(times, combined_data[:,1], label='Right Channel', color='darkorange')
            else:
                ax.plot(times, combined_data, color='steelblue')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            if params['channels'] == "Stereo":
                ax.legend()
            st.pyplot(fig_waveform)

            # Plot spectrum
            st.markdown("#### 📊 Frequency Spectrum")
            fig_spectrum, ax = plt.subplots()
            if params['channels'] == "Stereo":
                data_mono = combined_data.mean(axis=1)
            else:
                data_mono = combined_data.flatten()
            freqs = np.fft.rfftfreq(len(data_mono), 1/params['sample_rate'])
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
            img = librosa.display.specshow(D, sr=params['sample_rate'], x_axis='time', y_axis='log', ax=ax)
            ax.set_title('Spectrogram')
            fig_spectrogram.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig_spectrogram)

            st.markdown("---")  # Separator between samples

    else:
        st.write("Adjust parameters and click **Generate Noise** to create your industrial noise samples.")

if __name__ == "__main__":
    main()
