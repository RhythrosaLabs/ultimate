import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Industrial Noise Generator",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance appearance
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
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
st.title("🔊 Industrial Noise Generator")
st.markdown("""
Generate industrial noise samples at **48kHz mono**. Customize parameters to create unique sounds.

""")

# Sidebar for parameters
st.sidebar.header("🎛️ Controls")

duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=60, value=5)
noise_type = st.sidebar.selectbox("Noise Type", ["White Noise", "Pink Noise", "Brown Noise"])
lowcut = st.sidebar.slider("Low Cut Frequency (Hz)", min_value=20, max_value=1000, value=100)
highcut = st.sidebar.slider("High Cut Frequency (Hz)", min_value=1000, max_value=20000, value=5000)
order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=6)

# Functions to generate noise
def generate_white_noise(duration, sample_rate):
    samples = np.random.normal(0, 1, int(duration * sample_rate))
    return samples

def generate_pink_noise(duration, sample_rate):
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

# Main function
def main():
    if st.button("🎶 Generate Noise"):
        sample_rate = 48000
        
        # Generate noise based on selection
        if noise_type == "White Noise":
            data = generate_white_noise(duration, sample_rate)
        elif noise_type == "Pink Noise":
            data = generate_pink_noise(duration, sample_rate)
        elif noise_type == "Brown Noise":
            data = generate_brown_noise(duration, sample_rate)
        
        # Apply filter
        filtered_data = apply_filter(data, lowcut, highcut, sample_rate, order)
        
        # Normalize audio
        filtered_data = filtered_data / np.max(np.abs(filtered_data))
        
        # Save audio to buffer
        buffer = io.BytesIO()
        write(buffer, sample_rate, filtered_data.astype(np.float32))
        buffer.seek(0)
        
        # Play audio
        st.audio(buffer, format='audio/wav')
        
        # Provide download button
        st.download_button(label="💾 Download WAV", data=buffer, file_name="industrial_noise.wav", mime="audio/wav")
        
        # Plot waveform
        st.markdown("#### 📈 Waveform")
        fig_waveform, ax = plt.subplots()
        times = np.linspace(0, duration, len(filtered_data))
        ax.plot(times, filtered_data, color='steelblue')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig_waveform)
        
        # Plot spectrum
        st.markdown("#### 📊 Frequency Spectrum")
        fig_spectrum, ax = plt.subplots()
        freqs = np.fft.rfftfreq(len(filtered_data), 1/sample_rate)
        fft_magnitude = np.abs(np.fft.rfft(filtered_data))
        ax.semilogx(freqs, fft_magnitude, color='darkorange')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        st.pyplot(fig_spectrum)

# Run the app
if __name__ == "__main__":
    main()
