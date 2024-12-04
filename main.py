import streamlit as st
import numpy as np
import librosa
import random

# Initialize game state
if "game_state" not in st.session_state:
    st.session_state.game_state = "start"
    st.session_state.inventory = []
    st.session_state.completed_puzzles = []

# Helper functions
def analyze_audio(audio_input, target_pitch=None, target_beat=None):
    """
    Analyze the audio input for pitch or beat matching.
    """
    # Placeholder for pitch/beat analysis
    try:
        audio_data, sr = librosa.load(audio_input.name, sr=None)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        beat_times = librosa.beat.beat_track(y=audio_data, sr=sr)[1]

        # Check if the target pitch or beat is met (simplified)
        if target_pitch:
            if any(np.isclose(pitches, target_pitch, atol=50)):
                return "correct"
        if target_beat:
            if len(beat_times) >= target_beat:
                return "correct"
        return "incorrect"
    except Exception as e:
        return "incorrect"

def progress_story():
    if st.session_state.game_state == "start":
        st.session_state.game_state = "village"  # Progress to the next stage
    elif st.session_state.game_state == "village":
        st.session_state.game_state = "forest"
    elif st.session_state.game_state == "forest":
        st.session_state.game_state = "tower"

# Game scenes
def start_scene():
    st.title("Echoes of the Enigma")
    st.write("You awaken in the mystical land of Sonaria, where a haunting melody fills the air. Your journey begins at the crossroads. Do you choose to head towards the village or into the dark forest?")
    choice = st.radio("Choose your path:", ["Village", "Forest"])
    if st.button("Confirm Choice"):
        if choice == "Village":
            st.session_state.game_state = "village"
        elif choice == "Forest":
            st.session_state.game_state = "forest"

def village_scene():
    st.title("The Village")
    st.write("You arrive at a bustling village entranced by the melody. A villager approaches and says: \"The path to the tower is sealed by a harmonic gate. Can you match the tune?\"")
    st.audio("path_to_hint_audio/village_hint.mp3", format="audio/mp3")
    audio_input = st.audio_input("Sing the melody to unlock the gate")
    if audio_input:
        result = analyze_audio(audio_input, target_pitch=440)  # Example target pitch: 440Hz (A4)
        if result == "correct":
            st.success("The gate opens with a resonant chime!")
            progress_story()
        else:
            st.error("The melody doesn’t match. Try again!")

def forest_scene():
    st.title("The Forest")
    st.write("The forest is alive with whispers. A glowing tree speaks: \"I will grant you a key if you mimic my rhythm.\"")
    st.audio("path_to_hint_audio/forest_hint.mp3", format="audio/mp3")
    audio_input = st.audio_input("Clap or mimic the rhythm of the glowing tree")
    if audio_input:
        result = analyze_audio(audio_input, target_beat=3)  # Example target beat count
        if result == "correct":
            st.success("The tree hands you a glowing key!")
            st.session_state.inventory.append("Glowing Key")
            progress_story()
        else:
            st.error("The rhythm is incorrect. Listen closely and try again.")

def tower_scene():
    st.title("The Echo Tower")
    st.write("You stand before the towering spire. A final puzzle blocks your way: a sequence of tones to replicate.")
    st.audio("path_to_hint_audio/tower_hint.mp3", format="audio/mp3")
    audio_input = st.audio_input("Replicate the tones using your voice")
    if audio_input:
        result = analyze_audio(audio_input, target_pitch=523.25)  # Example target pitch: 523.25Hz (C5)
        if result == "correct":
            st.success("The tones resonate perfectly. The tower’s gate opens, revealing its mysteries!")
            st.balloons()
            st.write("Congratulations! You have unlocked the secrets of the Echo Tower and freed Sonaria from its enchantment.")
            st.session_state.game_state = "end"
        else:
            st.error("The tones don’t align. Try again.")

def end_scene():
    st.title("The End")
    st.write("Thank you for playing Echoes of the Enigma! Your voice has brought harmony to the land.")

# Game state manager
if st.session_state.game_state == "start":
    start_scene()
elif st.session_state.game_state == "village":
    village_scene()
elif st.session_state.game_state == "forest":
    forest_scene()
elif st.session_state.game_state == "tower":
    tower_scene()
elif st.session_state.game_state == "end":
    end_scene()
