import streamlit as st
import wave
import numpy as np

# Set the title of the Streamlit app
st.title("Audio Recorder and Playback")

# Provide a brief description of the app
st.write("Record your audio, play it back, and optionally save it as a file.")

# Display the audio input widget
st.write("### Step 1: Record your audio")
audio_data = st.audio_input("Please record your message:")

# If audio is recorded, play it back
if audio_data:
    st.write("### Step 2: Play back your recording")
    st.audio(audio_data, format='audio/wav')

    # Provide options for saving the audio and analyzing it
    st.write("### Step 3: Save or analyze your recording")

    save_option = st.checkbox("Save this recording")

    if save_option:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        st.success("Audio recording saved as 'recorded_audio.wav'")

    analyze_option = st.checkbox("Analyze this recording")

    if analyze_option:
        # Save audio data temporarily for analysis
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())

        # Perform audio analysis
        try:
            with wave.open("temp_audio.wav", "rb") as wav_file:
                frames = wav_file.readframes(-1)
                frame_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                st.write("#### Audio Analysis")
                st.write(f"**Duration:** {n_frames / frame_rate:.2f} seconds")
                st.write(f"**Sample Rate:** {frame_rate} Hz")
                st.write(f"**Channels:** {n_channels}")
                st.write(f"**Sample Width:** {sample_width} bytes")

                # Compute amplitude (for visualization purposes)
                audio_signal = np.frombuffer(frames, dtype=np.int16)
                st.line_chart(audio_signal[:min(1000, len(audio_signal))])

        except Exception as e:
            st.error(f"Error analyzing audio: {e}")
