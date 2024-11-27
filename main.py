import streamlit as st
import soundfile as sf
import numpy as np
import io
from scipy.signal import butter, lfilter, freqz
import librosa
import librosa.display
from pydub.effects import normalize
from pydub import AudioSegment
import random
import matplotlib.pyplot as plt
import logging
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Setup logging
logging.basicConfig(filename='audio_studio.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Utility functions for filtering audio
def process_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y, b, a

def process_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y, b, a

def add_reverb(data, decay_factor=0.5):
    reverb = np.zeros_like(data)
    for i in range(1, len(data)):
        reverb[i] = data[i] + decay_factor * reverb[i - 1]
    return reverb

def process_bitcrusher(data, bit_depth=8):
    max_amplitude = np.max(np.abs(data))
    step = max_amplitude / (2**bit_depth)
    crushed_audio = np.round(data / step) * step
    return crushed_audio

def process_chorus(data, samplerate):
    delay_samples = int(0.02 * samplerate)  # 20ms delay
    chorus = np.zeros_like(data)
    for i in range(delay_samples, len(data)):
        chorus[i] = data[i] + 0.5 * data[i - delay_samples]
    return chorus

def process_phaser(data, samplerate):
    phase_shift = np.sin(np.linspace(0, np.pi * 2, len(data)))
    return data * (1 + 0.5 * phase_shift)

def process_overdrive(data):
    return np.clip(data * 2, -1, 1)

# Additional Effects
def process_distortion(data, gain=1.0, threshold=0.3):
    distorted = data * gain
    distorted = np.where(distorted > threshold, threshold, distorted)
    distorted = np.where(distorted < -threshold, -threshold, distorted)
    return distorted

def process_flanger(data, samplerate, delay=0.005, depth=0.002, rate=0.25):
    delay_samples = int(delay * samplerate)
    depth_samples = int(depth * samplerate)
    flanger = np.zeros_like(data)
    for i in range(delay_samples, len(data)):
        index = i - delay_samples + depth_samples
        if index < 0:
            index = 0
        elif index >= len(data):
            index = len(data) - 1
        flanger[i] = data[i] + 0.7 * data[index]
    return flanger

def process_equalization(data, samplerate, gain_freqs=[(250, 500), (750, 1500), (2250, 3750)], gains=[1.5, 1.0, 0.8]):
    # Simple equalizer using band-pass filters
    eq_audio = data.copy()
    for (low, high), gain in zip(gain_freqs, gains):
        b, a = butter(2, [low / (0.5 * samplerate), high / (0.5 * samplerate)], btype='band')
        eq_audio += gain * lfilter(b, a, data)
    eq_audio = np.clip(eq_audio, -1, 1)
    return eq_audio

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
    },
    "Distorted Rock": {
        "apply_distortion_flag": True, "distortion_gain": 2.0, "distortion_threshold": 0.3,
        "apply_flanger_flag": True, "flanger_delay": 0.005, "flanger_depth": 0.002, "flanger_rate": 0.25,
    },
    "Equalized Pop": {
        "apply_equalization_flag": True, "eq_gain_freqs": [(300, 600), (1000, 2000), (3000, 6000)], "eq_gains": [1.5, 1.0, 0.8],
    }
}

# Define the AudioProcessor callback
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
    
    def recv(self, frame):
        audio = frame.to_ndarray().mean(axis=1)  # Convert to mono
        self.audio_frames.append(audio)
        return frame

# Define WebRTC client settings
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"audio": True, "video": False},
)

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

# Audio recording section using streamlit-webrtc
st.header("🎤 Record Your Audio")

# Initialize the audio processor
audio_processor = AudioProcessor()

# Start the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.RECVONLY,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    audio_receiver_size=256,
    async_processing=True,
    on_audio_frame=audio_processor.recv,
)

st.write("Press the microphone button to start recording. Once done, click the button below to process and download your audio.")

if st.button("Process and Download Audio"):
    try:
        if not audio_processor.audio_frames:
            st.warning("No audio recorded yet. Please record some audio first.")
        else:
            # Concatenate all recorded audio frames
            recorded_audio = np.concatenate(audio_processor.audio_frames)
            
            # Normalize audio
            recorded_audio = recorded_audio / np.max(np.abs(recorded_audio))
            
            # Convert to 16-bit PCM
            recorded_audio_int16 = (recorded_audio * 32767).astype(np.int16)
            
            # Create a BytesIO buffer
            buffer = io.BytesIO()
            sf.write(buffer, recorded_audio_int16, 44100, format='WAV')  # Adjust sample rate if necessary
            buffer.seek(0)
            
            # Provide download button
            st.download_button(
                label="Download WAV",
                data=buffer,
                file_name="recorded_audio.wav",
                mime="audio/wav"
            )
            
            st.success("Audio processed and ready for download!")
            
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        logging.error(f"Error processing audio: {e}")

# Check if an audio file is uploaded (optional)
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    try:
        audio_input = uploaded_file
        audio_data, samplerate = sf.read(audio_input)
        
        # Ensure mono and float32 format for compatibility
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        
        st.success("Audio successfully uploaded!")
        
        # Playback original audio
        st.subheader("Playback Original Audio")
        st.audio(audio_input, format='audio/wav')
        
        # User controls for effects
        st.sidebar.title("🎛️ Audio Effects")
        
        # Tooltips for effects
        st.sidebar.markdown("""
        **Effect Descriptions:**
        - **Lowpass Filter:** Allows frequencies below the cutoff to pass and attenuates higher frequencies.
        - **Highpass Filter:** Allows frequencies above the cutoff to pass and attenuates lower frequencies.
        - **Reverb:** Adds a sense of space by simulating reverberations.
        - **Bitcrusher:** Reduces the audio resolution, creating a lo-fi effect.
        - **Chorus:** Creates a richer sound by mixing delayed copies of the signal.
        - **Phaser:** Creates a sweeping effect by altering the phase of the signal.
        - **Overdrive:** Simulates the warm distortion of analog equipment.
        - **Distortion:** Adds heavy clipping for aggressive sound.
        - **Flanger:** Creates a swirling effect by mixing the signal with a delayed copy.
        - **Equalization:** Adjusts the balance of specific frequency bands.
        - **Echo:** Adds repeated delayed copies of the signal.
        - **Speed Change:** Alters the playback speed of the audio.
        - **Pitch Shift:** Changes the pitch without affecting the speed.
        - **Amplify Volume:** Increases or decreases the audio volume.
        - **Reverse Audio:** Reverses the audio playback direction.
        """)
        
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
            # Speed Change and Pitch Shift are commented out
            # apply_speed_change = False
            # speed_factor = 1.0
            # apply_pitch_shift = False
            # pitch_shift_steps = 0
            apply_amplify = False
            amplification_factor = 1.0
            apply_echo = False
            echo_delay = 500
            echo_decay = 0.5
            apply_chorus_effect = False
            apply_phaser_effect = False
            apply_overdrive_effect = False
            apply_distortion_flag = False
            distortion_gain = 1.0
            distortion_threshold = 0.3
            apply_flanger_flag = False
            flanger_delay = 0.005
            flanger_depth = 0.002
            flanger_rate = 0.25
            apply_equalization_flag = False
            eq_gain_freqs = [(300, 600), (1000, 2000), (3000, 6000)]
            eq_gains = [1.5, 1.0, 0.8]
    
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
    
            # Commented Out: Speed Change
            # apply_speed_change = st.sidebar.checkbox("Change Speed")
            # if apply_speed_change:
            #     speed_factor = st.sidebar.slider("Speed Factor", 0.5, 2.0, 1.0)
            # else:
            #     speed_factor = 1.0
    
            # Commented Out: Pitch Shift
            # apply_pitch_shift = st.sidebar.checkbox("Pitch Shift")
            # if apply_pitch_shift:
            #     pitch_shift_steps = st.sidebar.slider("Pitch Shift Steps", -12, 12, 0)
            # else:
            #     pitch_shift_steps = 0
    
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
            apply_distortion_flag = st.sidebar.checkbox("Apply Distortion")
            if apply_distortion_flag:
                distortion_gain = st.sidebar.slider("Distortion Gain", 1.0, 10.0, 2.0)
                distortion_threshold = st.sidebar.slider("Distortion Threshold", 0.1, 1.0, 0.3)
            else:
                distortion_gain = 1.0
                distortion_threshold = 0.3
    
            apply_flanger_flag = st.sidebar.checkbox("Add Flanger")
            if apply_flanger_flag:
                flanger_delay = st.sidebar.slider("Flanger Delay (s)", 0.001, 0.02, 0.005)
                flanger_depth = st.sidebar.slider("Flanger Depth (s)", 0.001, 0.01, 0.002)
                flanger_rate = st.sidebar.slider("Flanger Rate (Hz)", 0.1, 5.0, 0.25)
            else:
                flanger_delay = 0.005
                flanger_depth = 0.002
                flanger_rate = 0.25
    
            apply_equalization_flag = st.sidebar.checkbox("Add Equalization")
            if apply_equalization_flag:
                eq_gain_freqs_selected = st.sidebar.multiselect("Select Frequency Bands for EQ", [300, 1000, 3000, 6000, 12000], default=[300, 1000, 3000])
                eq_gains = []
                for freq in eq_gain_freqs_selected:
                    gain = st.sidebar.slider(f"Gain for {freq} Hz", 0.5, 2.0, 1.0, step=0.1)
                    eq_gains.append(gain)
                # Convert list to list of tuples for bandpass
                eq_gain_freqs = [(freq - 100, freq + 100) for freq in eq_gain_freqs_selected]
            else:
                eq_gain_freqs = [(300, 600), (1000, 2000), (3000, 6000)]
                eq_gains = [1.0, 1.0, 1.0]
    
        # Randomize effects
        st.sidebar.title("🎲 Randomize Effects")
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
    
            # Commented Out: Speed Change
            # apply_speed_change = random.random() < craziness
            # speed_factor = random.uniform(0.5, 2.0) if apply_speed_change else 1.0
    
            # Commented Out: Pitch Shift
            # apply_pitch_shift = random.random() < craziness
            # pitch_shift_steps = random.randint(-12, 12) if apply_pitch_shift else 0
    
            apply_amplify = random.random() < craziness
            amplification_factor = random.uniform(0.5, 3.0) if apply_amplify else 1.0
    
            apply_echo = random.random() < craziness
            echo_delay = random.randint(100, 2000) if apply_echo else 500
            echo_decay = random.uniform(0.1, 1.0) if apply_echo else 0.5
    
            apply_chorus_effect = random.random() < craziness
            apply_phaser_effect = random.random() < craziness
            apply_overdrive_effect = random.random() < craziness
    
            apply_distortion_flag = random.random() < craziness
            distortion_gain = random.uniform(1.0, 10.0) if apply_distortion_flag else 1.0
            distortion_threshold = random.uniform(0.1, 1.0) if apply_distortion_flag else 0.3
    
            apply_flanger_flag = random.random() < craziness
            flanger_delay = random.uniform(0.001, 0.02) if apply_flanger_flag else 0.005
            flanger_depth = random.uniform(0.001, 0.01) if apply_flanger_flag else 0.002
            flanger_rate = random.uniform(0.1, 5.0) if apply_flanger_flag else 0.25
    
            apply_equalization_flag = random.random() < craziness
            if apply_equalization_flag:
                selected_freqs = random.sample([300, 1000, 3000, 6000, 12000], random.randint(1, 3))
                eq_gain_freqs = [(freq - 100, freq + 100) for freq in selected_freqs]
                eq_gains = [random.uniform(0.5, 2.0) for _ in selected_freqs]
            else:
                eq_gain_freqs = [(300, 600), (1000, 2000), (3000, 6000)]
                eq_gains = [1.0, 1.0, 1.0]
    
            st.sidebar.success("Effects randomized!")
    
    # Apply effects
    if uploaded_file is not None:
        processed_audio = audio_data
    
        if apply_lowpass:
            try:
                processed_audio, b_low, a_low = process_lowpass_filter(processed_audio, cutoff_freq_low, samplerate)
                st.sidebar.success(f"Lowpass filter applied at {cutoff_freq_low} Hz")
            except Exception as e:
                st.sidebar.error(f"Error applying lowpass filter: {e}")
                logging.error(f"Error applying lowpass filter: {e}")
    
        if apply_highpass:
            try:
                processed_audio, b_high, a_high = process_highpass_filter(processed_audio, cutoff_freq_high, samplerate)
                st.sidebar.success(f"Highpass filter applied at {cutoff_freq_high} Hz")
            except Exception as e:
                st.sidebar.error(f"Error applying highpass filter: {e}")
                logging.error(f"Error applying highpass filter: {e}")
    
        if apply_reverb:
            try:
                processed_audio = add_reverb(processed_audio, reverb_decay)
                st.sidebar.success(f"Reverb added with decay factor {reverb_decay}")
            except Exception as e:
                st.sidebar.error(f"Error adding reverb: {e}")
                logging.error(f"Error adding reverb: {e}")
    
        if apply_bitcrusher:
            try:
                processed_audio = process_bitcrusher(processed_audio, bit_depth)
                st.sidebar.success(f"Bitcrusher applied with bit depth {bit_depth}")
            except Exception as e:
                st.sidebar.error(f"Error applying bitcrusher: {e}")
                logging.error(f"Error applying bitcrusher: {e}")
    
        if reverse_audio:
            try:
                processed_audio = processed_audio[::-1]
                st.sidebar.success("Audio reversed")
            except Exception as e:
                st.sidebar.error(f"Error reversing audio: {e}")
                logging.error(f"Error reversing audio: {e}")
    
        # Commented Out: Speed Change
        # if apply_speed_change:
        #     try:
        #         # Ensure processed_audio is a float32 numpy array
        #         processed_audio = librosa.effects.time_stretch(processed_audio, speed_factor)
        #         st.sidebar.success(f"Speed changed by a factor of {speed_factor}")
        #     except Exception as e:
        #         st.sidebar.error(f"Error changing speed: {e}")
        #         logging.error(f"Error changing speed: {e}")
    
        # Commented Out: Pitch Shift
        # if apply_pitch_shift:
        #     try:
        #         # Ensure processed_audio is a float32 numpy array
        #         processed_audio = librosa.effects.pitch_shift(processed_audio, samplerate, n_steps=pitch_shift_steps)
        #         st.sidebar.success(f"Pitch shifted by {pitch_shift_steps} semitones")
        #     except Exception as e:
        #         st.sidebar.error(f"Error applying pitch shift: {e}")
        #         logging.error(f"Error applying pitch shift: {e}")
    
        if apply_amplify:
            try:
                processed_audio = processed_audio * amplification_factor
                processed_audio = np.clip(processed_audio, -1.0, 1.0)
                st.sidebar.success(f"Volume amplified by a factor of {amplification_factor}")
            except Exception as e:
                st.sidebar.error(f"Error amplifying volume: {e}")
                logging.error(f"Error amplifying volume: {e}")
    
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
                logging.error(f"Error adding echo: {e}")
    
        if apply_chorus_effect:
            try:
                processed_audio = process_chorus(processed_audio, samplerate)
                st.sidebar.success("Chorus effect applied")
            except Exception as e:
                st.sidebar.error(f"Error applying chorus effect: {e}")
                logging.error(f"Error applying chorus effect: {e}")
    
        if apply_phaser_effect:
            try:
                processed_audio = process_phaser(processed_audio, samplerate)
                st.sidebar.success("Phaser effect applied")
            except Exception as e:
                st.sidebar.error(f"Error applying phaser effect: {e}")
                logging.error(f"Error applying phaser effect: {e}")
    
        if apply_overdrive_effect:
            try:
                processed_audio = process_overdrive(processed_audio)
                st.sidebar.success("Overdrive effect applied")
            except Exception as e:
                st.sidebar.error(f"Error applying overdrive effect: {e}")
                logging.error(f"Error applying overdrive effect: {e}")
    
        if apply_distortion_flag:
            try:
                processed_audio = process_distortion(processed_audio, gain=distortion_gain, threshold=distortion_threshold)
                st.sidebar.success(f"Distortion applied with gain {distortion_gain:.2f} and threshold {distortion_threshold:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error applying distortion: {e}")
                logging.error(f"Error applying distortion: {e}")
    
        if apply_flanger_flag:
            try:
                processed_audio = process_flanger(processed_audio, samplerate, delay=flanger_delay, depth=flanger_depth, rate=flanger_rate)
                st.sidebar.success(f"Flanger applied with delay {flanger_delay:.3f}s, depth {flanger_depth:.3f}s, and rate {flanger_rate:.2f}Hz")
            except Exception as e:
                st.sidebar.error(f"Error applying flanger: {e}")
                logging.error(f"Error applying flanger: {e}")
    
        if apply_equalization_flag:
            try:
                processed_audio = process_equalization(processed_audio, samplerate, gain_freqs=eq_gain_freqs, gains=eq_gains)
                st.sidebar.success(f"Equalization applied on frequency bands {eq_gain_freqs} Hz with gains {eq_gains}")
            except Exception as e:
                st.sidebar.error(f"Error applying equalization: {e}")
                logging.error(f"Error applying equalization: {e}")
    
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
            logging.error(f"Error normalizing audio: {e}")
    
        # Convert processed audio to bytes for playback and download
        try:
            buffer = io.BytesIO()
            sf.write(buffer, processed_audio, samplerate, format='WAV')
            buffer.seek(0)
        except Exception as e:
            st.error(f"Error preparing audio for playback/download: {e}")
            logging.error(f"Error preparing audio for playback/download: {e}")
            st.stop()
    
        # Live Preview
        if live_preview:
            st.subheader("🎧 Live Preview - Processed Audio")
            st.audio(buffer, format='audio/wav')
            buffer.seek(0)  # Reset buffer position after playback
    
        # Download processed audio
        st.subheader("💾 Download Processed Audio")
        st.download_button(
            label="Download WAV",
            data=buffer,
            file_name="processed_audio.wav",
            mime="audio/wav"
        )
    
        # Enhanced Visualization
        st.subheader("📊 Enhanced Visualization")
    
        # Waveform and Spectrogram Comparison
        try:
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
            # Original Waveform
            ax[0, 0].plot(audio_data, color='blue')
            ax[0, 0].set_title("Original Audio Waveform")
            ax[0, 0].set_xlabel("Samples")
            ax[0, 0].set_ylabel("Amplitude")
    
            # Processed Waveform
            ax[0, 1].plot(processed_audio, color='orange')
            ax[0, 1].set_title("Processed Audio Waveform")
            ax[0, 1].set_xlabel("Samples")
            ax[0, 1].set_ylabel("Amplitude")
    
            # Original Spectrogram
            S_orig = librosa.stft(audio_data)
            S_db_orig = librosa.amplitude_to_db(np.abs(S_orig), ref=np.max)
            img_orig = librosa.display.specshow(S_db_orig, sr=samplerate, x_axis='time', y_axis='log', ax=ax[1, 0])
            ax[1, 0].set_title("Original Audio Spectrogram")
            fig.colorbar(img_orig, ax=ax[1, 0], format="%+2.f dB")
    
            # Processed Spectrogram
            S_proc = librosa.stft(processed_audio)
            S_db_proc = librosa.amplitude_to_db(np.abs(S_proc), ref=np.max)
            img_proc = librosa.display.specshow(S_db_proc, sr=samplerate, x_axis='time', y_axis='log', ax=ax[1, 1])
            ax[1, 1].set_title("Processed Audio Spectrogram")
            fig.colorbar(img_proc, ax=ax[1, 1], format="%+2.f dB")
    
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating spectrogram visualization: {e}")
            logging.error(f"Error generating spectrogram visualization: {e}")
    
        # Frequency Response Plot for Filters
        if apply_lowpass or apply_highpass:
            try:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                if apply_lowpass:
                    w_low, h_low = freqz(b_low, a_low, worN=8000)
                    ax2.plot(0.5 * samplerate * w_low / np.pi, np.abs(h_low), label=f'Lowpass ({cutoff_freq_low} Hz)')
                if apply_highpass:
                    w_high, h_high = freqz(b_high, a_high, worN=8000)
                    ax2.plot(0.5 * samplerate * w_high / np.pi, np.abs(h_high), label=f'Highpass ({cutoff_freq_high} Hz)')
                ax2.set_title("Frequency Response")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Gain")
                ax2.legend()
                ax2.grid()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error generating frequency response plot: {e}")
                logging.error(f"Error generating frequency response plot: {e}")
    
else:
    st.info("📝 Please upload or record an audio file to get started.")
