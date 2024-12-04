import streamlit as st

# Set the title of the Streamlit app
st.title("Audio Recorder and Playback")

# Display the audio input widget
audio_data = st.audio_input("Please record your message:")

# If audio is recorded, play it back
if audio_data:
    st.audio(audio_data, format='audio/wav')

    # Provide an option to save the audio
    save_option = st.checkbox("Save this recording")

    if save_option:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_data.getbuffer())
        st.success("Audio recording saved as 'recorded_audio.wav'")
