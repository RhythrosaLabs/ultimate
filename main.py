import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import zipfile
import os
import replicate
import time

# Import custom modules (ensure these files are present in your directory)
try:
    from helpers import (
        get_api_keys, add_file_to_global_storage, generate_content, 
        generate_budget_spreadsheet, generate_social_media_schedule,
        generate_images, create_master_document, create_zip,
        enhance_content, add_to_chat_knowledge_base, create_gif,
        generate_audio_logo, generate_video_logo, animate_image_to_video,
        fetch_generated_video, generate_file_with_gpt
    )
except ModuleNotFoundError as e:
    st.error(f"Error importing helpers: {e}")
    
try:
    from image_tools import (
        mirror_image, invert_colors, pixelate_image, remove_background,
        greyscale_image, enhance_saturation, sepia_tone, blur_image,
        emboss_image, solarize_image, posterize_image, sharpen_image,
        apply_vhs_glitch_effect, apply_oil_painting_effect, flip_vertically,
        edge_detection, vignette, vintage_filter, thermal_vision,
        brighten_image, darken_image, rotate_left, rotate_right,
        dreamy_effect, glitch_art, pop_culture_filter, watercolor
    )
except ModuleNotFoundError as e:
    st.error(f"Error importing image_tools: {e}")

# Set up the page
st.set_page_config(page_title="AI-Powered Creative Suite", layout="wide")

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'global_file_storage' not in st.session_state:
    st.session_state.global_file_storage = {}

# Sidebar for API key input
with st.sidebar:
    st.title("API Key Setup")
    openai_key = st.text_input("OpenAI API Key", type="password")
    replicate_key = st.text_input("Replicate API Key", type="password")
    stability_key = st.text_input("Stability AI API Key", type="password")
    clipdrop_key = st.text_input("ClipDrop API Key", type="password")
    
    if st.button("Save API Keys"):
        st.session_state.api_keys = {
            "openai": openai_key,
            "replicate": replicate_key,
            "stability": stability_key,
            "clipdrop": clipdrop_key
        }
        st.success("API keys saved successfully!")

# Tabs for different functionality
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "File Management", "Image Editor", "Marketing Campaign Generator", 
    "Smart Notes", "Chat Assistant", "Custom Workflows"
])

# Tab 1: File Management
with tab1:
    st.header("File Management")
    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx'])
    if uploaded_file:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        file_contents = uploaded_file.read()
        try:
            add_file_to_global_storage(uploaded_file.name, file_contents)
            st.success(f"File {uploaded_file.name} uploaded and stored.")
        except Exception as e:
            st.error(f"Failed to upload file: {e}")
    
    if st.button("View Stored Files"):
        for filename, content in st.session_state.global_file_storage.items():
            st.write(f"- {filename}")

# Tab 2: Image Editor
with tab2:
    st.header("Image Editor")
    image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        effect = st.selectbox("Choose an effect", [
            "Mirror", "Invert Colors", "Pixelate", "Greyscale", "Enhance Saturation",
            "Sepia", "Blur", "Emboss", "Solarize", "Posterize", "Sharpen",
            "VHS Glitch", "Oil Painting", "Flip Vertically", "Edge Detection",
            "Vignette", "Vintage", "Thermal Vision", "Brighten", "Darken",
            "Rotate Left", "Rotate Right", "Dreamy", "Glitch Art", "Pop Culture", "Watercolor"
        ])
        
        if st.button("Apply Effect"):
            try:
                effect_function = globals()[effect.lower().replace(" ", "_") + "_image"]
                edited_image = effect_function(image)
                st.image(edited_image, caption="Edited Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error applying effect: {e}")

# Tab 3: Marketing Campaign Generator
with tab3:
    st.header("Marketing Campaign Generator")
    campaign_prompt = st.text_area("Describe your marketing campaign:")
    budget = st.number_input("Budget ($)", min_value=100, value=1000)
    
    if st.button("Generate Campaign"):
        try:
            campaign_plan = {}
            campaign_plan['campaign_concept'] = generate_content("Generate campaign concept", campaign_prompt, str(budget), {}, st.session_state.api_keys['openai'])
            campaign_plan['marketing_plan'] = generate_content("Generate marketing plan", campaign_prompt, str(budget), {}, st.session_state.api_keys['openai'])
            campaign_plan['budget_spreadsheet'] = generate_budget_spreadsheet(budget)
            campaign_plan['social_media_schedule'] = generate_social_media_schedule(campaign_plan['campaign_concept'], {"facebook": True, "twitter": True, "instagram": True})
            
            master_document = create_master_document(campaign_plan)
            zip_data = create_zip(campaign_plan)
            
            st.download_button(
                label="Download Campaign ZIP",
                data=zip_data.getvalue(),
                file_name="marketing_campaign.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Error generating campaign: {e}")

# Tab 4: Smart Notes
with tab4:
    st.header("Smart Notes")
    note = st.text_area("Type your notes here...")
    if st.button("Enhance Notes"):
        try:
            enhanced_note = enhance_content(note, "Smart Note")
            st.write("Enhanced Note:")
            st.write(enhanced_note)
        except Exception as e:
            st.error(f"Error enhancing notes: {e}")

# Tab 5: Chat Assistant
with tab5:
    st.header("Chat Assistant")
    user_message = st.text_input("Ask me anything:")
    if st.button("Send"):
        try:
            response = generate_content("Chat response", user_message, "", {}, st.session_state.api_keys['openai'])
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error during chat: {e}")
    
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

# Tab 6: Custom Workflows
with tab6:
    st.header("Custom Workflows")
    workflow_prompt = st.text_area("Describe your custom workflow:")
    if st.button("Generate Workflow"):
        try:
            workflow_files = generate_file_with_gpt(workflow_prompt)
            if workflow_files:
                st.success("Workflow generated successfully!")
                for filename, content in workflow_files.items():
                    st.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/plain"
                    )
        except Exception as e:
            st.error(f"Error generating workflow: {e}")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by AI")
