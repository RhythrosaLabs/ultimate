import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import matplotlib.pyplot as plt
import io
import librosa
import librosa.display
import random

# Set page configuration
st.set_page_config(
    page_title="Industrial Noise Generator Pro",
    page_icon="🔊",
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
st.title("🔊 Industrial Noise Generator Pro")
st.markdown("""
Generate industrial noise samples at **48kHz mono** or customize your sample rate. Customize parameters to create unique sounds.
""")

# Sidebar for parameters
st.sidebar.header("🎛️ Controls")

# Define noise_options here, outside of any conditional blocks
noise_options = ["White Noise", "Pink Noise", "Brown Noise", "Blue Noise", "Violet Noise", "Grey Noise"]

# Presets
st.sidebar.subheader("🎚️ Presets")
preset_options = ["Default", "Heavy Machinery", "Factory Floor", "Electric Hum", "Custom"]
preset = st.sidebar.selectbox("Choose a Preset", preset_options)

# Function to set preset parameters
def set_preset(preset):
    params = {}
    if preset == "Default":
        params = {
            'duration': 5,
            'noise_type': ["White Noise"],
            'lowcut': 100,
            'highcut': 5000,
            'order': 6,
            'amplitude': 1.0,
            'sample_rate': 48000,
            'channels': "Mono",
            'fade_in': 0.0,
            'fade_out': 0.0,
            'reverb': False,
            'delay': False,
            'distortion': False,
            'panning': 0.5,
            'bit_depth': 16,
            'modulation': None
        }
    elif preset == "Heavy Machinery":
        params = {
            'duration': 10,
            'noise_type': ["Brown Noise", "Pink Noise"],
            'lowcut': 50,
            'highcut': 2000,
            'order': 8,
            'amplitude': 1.0,
            'sample_rate': 48000,
            'channels': "Stereo",
            'fade_in': 1.0,
            'fade_out': 1.0,
            'reverb': True,
            'delay': True,
            'distortion': True,
            'panning': 0.5,
            'bit_depth': 16,
            'modulation': "Amplitude Modulation"
        }
    elif preset == "Factory Floor":
        params = {
            'duration': 8,
            'noise_type': ["White Noise", "Brown Noise"],
            'lowcut': 150,
            'highcut': 4000,
            'order': 4,
            'amplitude': 0.8,
            'sample_rate': 48000,
            'channels': "Mono",
            'fade_in': 0.5,
            'fade_out': 0.5,
            'reverb': True,
            'delay': False,
            'distortion': True,
            'panning': 0.0,
            'bit_depth': 24,
            'modulation': None
        }
    elif preset == "Electric Hum":
        params = {
            'duration': 5,
            'noise_type': ["Violet Noise"],
            'lowcut': 5000,
            'highcut': 20000,
            'order': 6,
            'amplitude': 0.6,
            'sample_rate': 48000,
            'channels': "Mono",
            'fade_in': 0.2,
            'fade_out': 0.2,
            'reverb': False,
            'delay': False,
            'distortion': False,
            'panning': 0.5,
            'bit_depth': 16,
            'modulation': "Frequency Modulation"
        }
    else:  # Custom
        params = None
    return params

preset_params = set_preset(preset)

if preset != "Custom" and preset_params is not None:
    # Set parameters from preset
    duration = preset_params['duration']
    noise_types = preset_params['noise_type']
    lowcut = preset_params['lowcut']
    highcut = preset_params['highcut']
    order = preset_params['order']
    amplitude = preset_params['amplitude']
    sample_rate = preset_params['sample_rate']
    channels = preset_params['channels']
    fade_in = preset_params['fade_in']
    fade_out = preset_params['fade_out']
    reverb = preset_params['reverb']
    delay = preset_params['delay']
    distortion = preset_params['distortion']
    panning = preset_params['panning']
    bit_depth = preset_params['bit_depth']
    modulation = preset_params['modulation']
else:
    # Custom parameters
    duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=5)
    noise_types = st.sidebar.multiselect("Noise Types", noise_options, default=["White Noise"])
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
    st.sidebar.subheader("🎵 Effects")
    reverb = st.sidebar.checkbox("Add Reverb")
    delay = st.sidebar.checkbox("Add Delay")
    distortion = st.sidebar.checkbox("Add Distortion")
    modulation = st.sidebar.selectbox("Modulation", [None, "Amplitude Modulation", "Frequency Modulation"])

# Randomize button
if st.sidebar.button("🔀 Randomize Parameters"):
    duration = random.randint(1, 60)
    noise_types = random.sample(noise_options, random.randint(1, len(noise_options)))
    lowcut = random.randint(20, 10000)
    highcut = random.randint(lowcut + 100, 24000)
    order = random.randint(1, 10)
    amplitude = random.uniform(0.0, 1.0)
    sample_rate = random.choice([48000, 44100, 32000, 22050, 16000, 8000])
    channels = random.choice(["Mono", "Stereo"])
    fade_in = random.uniform(0.0, 5.0)
    fade_out = random.uniform(0.0, 5.0)
    panning = random.uniform(0.0, 1.0)
    bit_depth = random.choice([16, 24, 32])
    reverb = random.choice([True, False])
    delay = random.choice([True, False])
    distortion = random.choice([True, False])
    modulation = random.choice([None, "Amplitude Modulation", "Frequency Modulation"])

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
    brown_noise = brown_noise / np.max(np.abs(brown_noise))
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
    a_weighting = a_weighting / np.max(a_weighting)
    white_fft = np.fft.rfft(white)
    grey_fft = white_fft * a_weighting
    grey_noise = np.fft.irfft(grey_fft)
    return grey_noise

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
    # Apply fade in/out
    total_samples = len(data)
    fade_in_samples = int(fade_in * sample_rate)
    fade_out_samples = int(fade_out * sample_rate)
    fade_in_curve = np.linspace(0, 1, fade_in_samples)
    fade_out_curve = np.linspace(1, 0, fade_out_samples)
    data[:fade_in_samples] *= fade_in_curve
    data[-fade_out_samples:] *= fade_out_curve
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

# Main function
def main():
    if st.button("🎶 Generate Noise"):
        # Generate noise based on selection
        combined_data = np.zeros(int(duration * sample_rate))

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
                data = np.zeros(int(duration * sample_rate))
            
            # Apply filter
            data = apply_filter(data, lowcut, highcut, sample_rate, order)
            
            # Normalize audio
            data = data / np.max(np.abs(data) + 1e-7)  # Added epsilon to prevent division by zero
            
            # Combine noises
            combined_data += data

        # Normalize combined data
        combined_data = combined_data / np.max(np.abs(combined_data) + 1e-7)
        
        # Apply amplitude
        combined_data *= amplitude

        # Apply fade in/out
        combined_data = apply_fade(combined_data, sample_rate, fade_in, fade_out)

        # Apply modulation
        if modulation == "Amplitude Modulation":
            combined_data = apply_amplitude_modulation(combined_data, sample_rate)
        elif modulation == "Frequency Modulation":
            combined_data = apply_frequency_modulation(combined_data, sample_rate)

        # Apply effects
        if reverb:
            combined_data = apply_reverb(combined_data, sample_rate)
        if delay:
            combined_data = apply_delay(combined_data, sample_rate)
        if distortion:
            combined_data = apply_distortion(combined_data)

        # Adjust bit depth
        combined_data = adjust_bit_depth(combined_data, bit_depth)

        # Handle stereo or mono
        if channels == "Stereo":
            combined_data = pan_stereo(combined_data, panning)
        else:
            combined_data = combined_data.reshape(-1, 1)

        # Convert to proper dtype for saving
        if bit_depth == 16:
            dtype = np.int16
            max_int = np.iinfo(dtype).max
            combined_data = combined_data * max_int
            combined_data = combined_data.astype(dtype)
        elif bit_depth == 24:
            # 24-bit WAV files are not standard; we'll use 32-bit float as a workaround
            dtype = np.float32
            combined_data = combined_data.astype(dtype)
        else:  # 32-bit
            dtype = np.float32
            combined_data = combined_data.astype(dtype)

        # Save audio to buffer
        buffer = io.BytesIO()
        write(buffer, sample_rate, combined_data)
        buffer.seek(0)
        
        # Play audio
        st.audio(buffer, format='audio/wav')
        
        # Provide download button
        st.download_button(label="💾 Download WAV", data=buffer, file_name="industrial_noise.wav", mime="audio/wav")
        
        # Plot waveform
        st.markdown("#### 📈 Waveform")
        fig_waveform, ax = plt.subplots()
        times = np.linspace(0, duration, len(combined_data))
        if channels == "Stereo":
            ax.plot(times, combined_data[:,0], label='Left Channel', color='steelblue')
            ax.plot(times, combined_data[:,1], label='Right Channel', color='darkorange')
        else:
            ax.plot(times, combined_data, color='steelblue')
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
            data_mono = combined_data.mean(axis=1)
        else:
            data_mono = combined_data.flatten()
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
    
    else:
        st.write("Adjust parameters and click **Generate Noise** to create your industrial noise sample.")

# Run the app
if __name__ == "__main__":
    main()
