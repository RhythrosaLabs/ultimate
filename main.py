import streamlit as st
import requests
import tempfile
from transformers import pipeline
from langdetect import detect
from io import BytesIO

# Set a custom page configuration for a sleek appearance
st.set_page_config(
    page_title="Rhyme Bot: Audio Rhymes with a Personality Twist!",
    page_icon="🎤",
    layout="centered"
)

# Custom CSS for a more stylish UI
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #d8b4fe, #90cdf4);
    font-family: 'Helvetica', sans-serif;
    color: #2D3748;
}

.header-title {
    text-align: center;
    color: #2D3748;
    font-weight: 900;
    font-size: 3em;
    margin-top: 0.5em;
    margin-bottom: 0.1em;
    text-shadow: 1px 1px #fff;
}

.sub-title {
    text-align: center;
    color: #4A5568;
    font-size: 1.2em;
    margin-bottom: 1.5em;
}

.prompt-label {
    font-size: 1.1em;
    font-weight: bold;
    color: #2D3748;
}

.warning {
    background-color: #fbd38d;
    padding: 0.5em;
    border-radius: 0.5em;
    font-weight: bold;
    color: #553C16;
    margin-top: 1em;
}

.personality-container {
    border: 2px solid #805AD5;
    border-radius: 0.5em;
    padding: 1em;
    background-color: #FAF5FF;
    margin: 1em 0;
}

.personality-header {
    font-weight: bold;
    font-size: 1.3em;
    color: #5A46A3;
    margin-bottom: 0.5em;
    text-align: center;
}

audio {
    margin-top: 1em;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# Main titles
st.markdown('<div class="header-title">Rhyme Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">🎶 Speak into the mic, and watch me transform your words into lyrical art! 🎶</div>', unsafe_allow_html=True)

# Instructions and API key input
st.markdown("""<span class="prompt-label">Enter your OpenAI API key:</span>""", unsafe_allow_html=True)
api_key = st.text_input(
    "",
    type="password",
    placeholder="Your OpenAI API key here..."
)

if not api_key:
    st.markdown('<div class="warning">🔑 Please enter your OpenAI API key to proceed.</div>', unsafe_allow_html=True)
else:
    # Personalities and associated prompts
    st.markdown('<div class="personality-container">', unsafe_allow_html=True)
    st.markdown('<div class="personality-header">Choose a Personality</div>', unsafe_allow_html=True)
    personality = st.radio(
        "",
        ("Shakespearean Poet", "Modern Rapper", "Playful Jokester"),
        index=1
    )
    st.markdown('</div>', unsafe_allow_html=True)

    persona_prompts = {
        "Shakespearean Poet": "Respond with a rhyming couplet in the style of Shakespeare.",
        "Modern Rapper": "Drop a rap line with rhyme and rhythm influenced by contemporary hip-hop.",
        "Playful Jokester": "Respond with a fun, playful rhyme sure to spark a grin."
    }

    system_prompt = persona_prompts[personality]

    # Dynamic instructions based on personality
    instructions = {
        "Shakespearean Poet": "🎭 Summon your inner bard and recite a phrase!",
        "Modern Rapper": "🎧 Speak your truth – I’ll turn it into a lyrical masterpiece!",
        "Playful Jokester": "🤡 Say something silly, and I'll spin it into a goofy rhyme!"
    }
    st.markdown(f"<span class='prompt-label'>{instructions[personality]}</span>", unsafe_allow_html=True)

    # Audio input for user's spoken words
    audio_file = st.audio_input("Your microphone awaits...")

    if audio_file:
        # Show recorded audio
        st.audio(audio_file, format="audio/wav")

        # Save the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        # Try transcription
        try:
            with st.spinner("🕰️ Transcribing your voice, please wait..."):
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

                    st.success("✅ Transcription completed!")
                    st.markdown("**You said:**")
                    st.write(f"> *{transcription}*")

                    # Detect language
                    language = detect(transcription)
                    if language != "en":
                        system_prompt = f"Respond with a rhyme in {language}."

                    # Analyze sentiment
                    sentiment_analyzer = pipeline("sentiment-analysis")
                    sentiment = sentiment_analyzer(transcription)[0]["label"]

                    # Tone adjustment based on sentiment
                    if sentiment == "NEGATIVE":
                        tone = "uplifting and encouraging"
                    elif sentiment == "POSITIVE":
                        tone = "joyful and celebratory"
                    else:
                        tone = "calm and neutral"

                    system_prompt += f" Use a tone that is {tone}."

                    # Request rhyme from GPT
                    with st.spinner("🎶 Crafting your perfect rhyme..."):
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
                            st.markdown("**Rhyme Bot responds:**")
                            st.markdown(f"> *{rhyming_line}*")
                        else:
                            st.error(
                                f"❌ Failed to generate rhyme: {response.status_code} {response.text}"
                            )
                else:
                    st.error(
                        f"❌ Failed to transcribe audio: {response.status_code} {response.text}"
                    )
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.markdown("""
        <span class="prompt-label">
        🎙️ Click "Record" above, say something, and let Rhyme Bot work its magic!
        </span>
        """, unsafe_allow_html=True)
