import streamlit as st
import random

# Initialize game state
if "game_state" not in st.session_state:
    st.session_state.game_state = "start"
    st.session_state.inventory = []
    st.session_state.completed_puzzles = []

# Helper functions
def analyze_audio(audio_input):
    """
    Placeholder for audio analysis logic.
    For example, return "correct" if the player mimics a required pattern correctly.
    """
    # Simulate random success/failure for now
    return random.choice(["correct", "incorrect"])

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
    audio_input = st.audio_input("Sing the melody to unlock the gate")
    if audio_input:
        result = analyze_audio(audio_input)
        if result == "correct":
            st.success("The gate opens with a resonant chime!")
            progress_story()
        else:
            st.error("The melody doesn’t match. Try again!")

def forest_scene():
    st.title("The Forest")
    st.write("The forest is alive with whispers. A glowing tree speaks: \"I will grant you a key if you mimic my rhythm.\"")
    audio_input = st.audio_input("Clap or mimic the rhythm of the glowing tree")
    if audio_input:
        result = analyze_audio(audio_input)
        if result == "correct":
            st.success("The tree hands you a glowing key!")
            st.session_state.inventory.append("Glowing Key")
            progress_story()
        else:
            st.error("The rhythm is incorrect. Listen closely and try again.")

def tower_scene():
    st.title("The Echo Tower")
    st.write("You stand before the towering spire. A final puzzle blocks your way: a sequence of tones to replicate.")
    st.write("The tones play: 🎵 Do-Re-Mi-Fa 🎵")
    audio_input = st.audio_input("Replicate the tones using your voice")
    if audio_input:
        result = analyze_audio(audio_input)
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
