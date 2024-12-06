import streamlit as st
import requests
import tempfile
from transformers import pipeline
from langdetect import detect
import json

# --------------------- Helper Functions --------------------- #

def get_rhyme_suggestions(word):
    # Placeholder: In a real implementation, call a rhyme API.
    # Example static suggestions:
    suggestions = {
        "day": ["play", "say", "way", "clay", "array"],
        "amor": ["dolor", "calor", "motor"],
    }
    return suggestions.get(word.lower(), ["No suggestions found."])

def generate_ai_image(prompt):
    # Placeholder for DALL·E image generation call.
    # If you have an API endpoint:
    # headers = {"Authorization": f"Bearer {api_key}"}
    # json_data = {"prompt": prompt, "n":1, "size":"512x512"}
    # response = requests.post("DALL_E_API_ENDPOINT", headers=headers, json=json_data)
    # image_url = response.json()["data"][0]["url"]
    # return image_url
    return None

def get_background_beat(tone):
    # Placeholder: return a URL or local file path to an audio file
    # In reality, you might have different audio files depending on tone.
    if "uplifting" in tone or "joyful" in tone:
        return "https://example.com/happy_beat.mp3"
    elif "encouraging" in tone:
        return "https://example.com/encouraging_beat.mp3"
    elif "calm" in tone:
        return "https://example.com/calm_beat.mp3"
    else:
        return "https://example.com/default_beat.mp3"

def construct_system_prompt(personality, mode, language, sentiment_tone):
    persona_prompts = {
        "Shakespearean Poet": "Respond with a rhyming couplet in the style of Shakespeare.",
        "Modern Rapper": "Drop a rap line with a rhyme and rhythm based on the user's input.",
        "Playful Jokester": "Respond with a rhyming line that's humorous and playful."
    }

    # Base persona prompt
    system_prompt = persona_prompts[personality]

    # Multilingual support
    if language != "en":
        system_prompt = f"Respond with a rhyme in {language}."

    # Adjust tone based on sentiment
    system_prompt += f" Respond in a tone that is {sentiment_tone}."

    # Additional mode instructions
    if mode == "Rhyme Battle":
        system_prompt += " We are in a rhyme battle. The user and I take turns adding lines that rhyme with each other."
    elif mode == "Storytelling":
        system_prompt += " We are creating a longer rhyming story, each of my responses should advance the narrative."
    elif mode == "Song Lyrics":
        system_prompt += " We are creating a song. The user provides verses, and I provide a rhyming chorus."

    return system_prompt


# --------------------- Streamlit App --------------------- #

st.title("Rhyme Bot: Speak, and I'll Rhyme!")

# Input for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="Enter your API key here...",
)

if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    # Personality Selector
    personality = st.radio(
        "Choose a personality for Rhyme Bot:",
        ("Shakespearean Poet", "Modern Rapper", "Playful Jokester")
    )

    # Advanced Features
    mode = st.selectbox(
        "Select mode:",
        ("Normal", "Rhyme Battle", "Storytelling", "Song Lyrics")
    )

    # Educational Features Toggle
    show_education = st.checkbox("Show Rhyme Tutorials & Word Suggestions")

    # Record audio
    audio_file = st.audio_input("Say something, and I'll rhyme!")

    if audio_file:
        # Display the recorded audio for playback
        st.audio(audio_file, format="audio/wav")

        # Save the audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper API
        try:
            with st.spinner("Transcribing your voice..."):
                with open(temp_audio_path, "rb") as audio:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                    }
                    files = {
                        "file": audio,
                    }
                    data = {
                        "model": "whisper-1",
                    }
                    response = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                    )
                if response.status_code == 200:
                    transcription = response.json()["text"]
                    st.success("Transcription completed!")
                    st.write("**You said:**")
                    st.write(transcription)

                    # Language Detection
                    language = detect(transcription)

                    # Sentiment Analysis
                    sentiment_analyzer = pipeline("sentiment-analysis")
                    sentiment = sentiment_analyzer(transcription)[0]["label"].upper()

                    # Determine tone based on sentiment
                    if sentiment == "NEGATIVE":
                        tone = "uplifting and encouraging"
                    elif sentiment == "POSITIVE":
                        tone = "joyful and celebratory"
                    else:
                        tone = "neutral and calm"

                    # Construct dynamic system prompt
                    system_prompt = construct_system_prompt(personality, mode, language, tone)

                    # Generate a rhyming response using GPT Chat API
                    with st.spinner("Let me rhyme..."):
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        }
                        chat_data = {
                            "model": "gpt-4",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": system_prompt,
                                },
                                {
                                    "role": "user",
                                    "content": transcription,
                                },
                            ],
                            "temperature": 0.8,
                        }
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=chat_data,
                        )
                        if response.status_code == 200:
                            rhyming_line = response.json()["choices"][0]["message"]["content"].strip()
                            st.write("**Rhyme Bot says:**")
                            
                            # Display rhyme with creative typography (placeholder)
                            # In practice, you could use st.markdown with custom CSS or images.
                            st.markdown(f"<h2 style='font-family:serif; color:#3A3A3A;'>{rhyming_line}</h2>", unsafe_allow_html=True)
                            
                            # Add background beat (if any)
                            beat_url = get_background_beat(tone)
                            if beat_url:
                                st.audio(beat_url, format="audio/mp3")

                            # AI-generated image based on the rhyme content (if available)
                            # For example, if user said something about a "sunny day":
                            if "sun" in transcription.lower():
                                image_prompt = "A bright sunny landscape with vibrant colors"
                                image_url = generate_ai_image(image_prompt)
                                if image_url:
                                    st.image(image_url, caption="AI-generated illustration")
                            
                            # Educational Features
                            if show_education:
                                st.write("**Rhyme Tutorial:**")
                                st.write("A common rhyme scheme is AABB, where the first two lines rhyme and the next two lines rhyme. For example:\nLine 1 (A) and Line 2 (A) rhyme.\nLine 3 (B) and Line 4 (B) rhyme.")
                                # Suggest rhymes for a word if the user wants
                                if transcription:
                                    last_word = transcription.strip().split()[-1]
                                    suggestions = get_rhyme_suggestions(last_word)
                                    st.write(f"**Word Suggestions for '{last_word}':**", suggestions)

                        else:
                            st.error(
                                f"Failed to generate rhyme: {response.status_code} {response.text}"
                            )
                else:
                    st.error(
                        f"Failed to transcribe audio: {response.status_code} {response.text}"
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
