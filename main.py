import streamlit as st
import numpy as np
from pydub import AudioSegment, effects
import matplotlib.pyplot as plt
from io import BytesIO

def change_speed(sound, speed=1.0):
    # Change the playback speed of the sound
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

# App Title
st.title("Supercharged Multi-Track Audio Editor")

# Sidebar for Track Upload
st.sidebar.header("Upload Tracks")
num_tracks = st.sidebar.slider("Number of Tracks", 1, 5, 2)

tracks = []
track_names = []

for i in range(num_tracks):
    st.sidebar.write(f"### Track {i + 1}")
    audio_file = st.sidebar.file_uploader(f"Upload Track {i + 1}", type=["wav", "mp3", "ogg"], key=f"file_{i}")
    if audio_file:
        track = AudioSegment.from_file(audio_file)
        tracks.append(track)
        track_names.append(f"Track {i + 1}")

# Mixing Interface
if tracks:
    st.header("Mixing Desk")
    st.write("Adjust volume, pan, apply effects, and trim each track.")

    mixed_audio = None

    for i, track in enumerate(tracks):
        st.subheader(track_names[i])

        # Volume control
        volume = st.slider(f"Volume (dB) - {track_names[i]}", -60.0, 10.0, 0.0, 0.1)

        # Pan control
        pan = st.slider(f"Pan - {track_names[i]}", -1.0, 1.0, 0.0, 0.01)

        # Start and End Time (Trim)
        duration_ms = len(track)
        st.write(f"Track Duration: {duration_ms/1000:.2f}s")
        start_time = st.number_input(f"Start Time (s) - {track_names[i]}", min_value=0.0, max_value=duration_ms/1000, value=0.0, step=0.1)
        end_time = st.number_input(f"End Time (s) - {track_names[i]}", min_value=0.0, max_value=duration_ms/1000, value=duration_ms/1000, step=0.1)

        # Effects
        effect_options = st.multiselect(f"Effects - {track_names[i]}", ['Normalize', 'Reverse', 'Speed Up', 'Slow Down'])

        # Waveform Visualization
        st.write("Waveform Preview:")
        trimmed_track = track[start_time * 1000:end_time * 1000]
        samples = np.array(trimmed_track.get_array_of_samples())
        fig, ax = plt.subplots()
        ax.plot(samples)
        st.pyplot(fig)

        # Apply volume and pan
        adjusted_track = trimmed_track + volume
        adjusted_track = adjusted_track.pan(pan)

        # Apply effects
        if 'Normalize' in effect_options:
            adjusted_track = effects.normalize(adjusted_track)
        if 'Reverse' in effect_options:
            adjusted_track = adjusted_track.reverse()
        if 'Speed Up' in effect_options:
            adjusted_track = change_speed(adjusted_track, 1.5)
        if 'Slow Down' in effect_options:
            adjusted_track = change_speed(adjusted_track, 0.75)

        # Mix tracks
        if mixed_audio:
            mixed_audio = mixed_audio.overlay(adjusted_track)
        else:
            mixed_audio = adjusted_track

    # Playback and Export
    st.header("Output")

    if st.button("Play Mixed Audio"):
        st.write("Playing Mixed Audio...")
        # Export to BytesIO
        mixed_audio_io = BytesIO()
        mixed_audio.export(mixed_audio_io, format="wav")
        mixed_audio_io.seek(0)
        st.audio(mixed_audio_io.read())

    if st.button("Export Mixed Audio"):
        st.write("Exporting...")
        # Export to BytesIO
        mixed_audio_io = BytesIO()
        mixed_audio.export(mixed_audio_io, format="wav")
        mixed_audio_io.seek(0)
        st.download_button("Download Mixed Audio", mixed_audio_io, file_name="mixed_output.wav")

    st.write("Enjoy your supercharged audio mixing experience!")

else:
    st.write("Please upload at least one track to start mixing.")
