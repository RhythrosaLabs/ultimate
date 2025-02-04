import streamlit as st
import requests
import json

# Gradio backend URL
GRADIO_API_URL = "http://localhost:7788/update_settings"

st.title("Dynamic Browser Automation Settings")

# User-adjustable parameters
timeout = st.slider("Timeout (seconds)", 1, 60, 10)
resolution = st.selectbox("Screen Resolution", ["1920x1080", "1280x720", "1024x768"])
headless = st.checkbox("Run Browser in Headless Mode", True)
auto_scroll = st.checkbox("Enable Auto Scrolling", False)
custom_script = st.text_area("Custom JavaScript to Inject", "")

# Create a settings dictionary
settings = {
    "timeout": timeout,
    "resolution": resolution,
    "headless": headless,
    "auto_scroll": auto_scroll,
    "custom_script": custom_script
}

# Send settings to Gradio backend
if st.button("Update Settings"):
    response = requests.post(GRADIO_API_URL, json=settings)
    if response.status_code == 200:
        st.success("Settings updated successfully!")
    else:
        st.error("Failed to update settings.")

# Display current settings
st.subheader("Current Settings")
st.json(settings)
