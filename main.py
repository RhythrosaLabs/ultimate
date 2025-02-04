import streamlit as st
import subprocess
import time
import requests
from gradio_client import Client

# --------------------- CONFIG --------------------- #
GRADIO_SERVER_URL = "http://localhost:7788/"
GRADIO_APP_PATH = "your_gradio_app.py"  # Change this to your actual Gradio app script path
GRADIO_PORT = 7788

# --------------------- HELPER FUNCTIONS --------------------- #
def is_gradio_running():
    """Check if Gradio is running on the specified port."""
    try:
        response = requests.get(GRADIO_SERVER_URL, timeout=3)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def start_gradio():
    """Start the Gradio server if it's not running."""
    if not is_gradio_running():
        st.warning("Starting Gradio server...")
        process = subprocess.Popen(
            ["python", GRADIO_APP_PATH, "--server-name", "0.0.0.0", "--server-port", str(GRADIO_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(5)  # Give the server time to start
        if is_gradio_running():
            st.success("Gradio server started successfully!")
        else:
            st.error("Failed to start Gradio server. Check your Gradio script.")
    else:
        st.success("Gradio server is already running.")

# --------------------- STREAMLIT UI --------------------- #
st.title("Auto-Configured Gradio Client")

# Start Gradio if not running
start_gradio()

# Connect to Gradio Client
if is_gradio_running():
    client = Client(GRADIO_SERVER_URL)
    
    # Example API Call (Modify as needed)
    try:
        result = client.predict(
            agent_type="custom",
            llm_provider="openai",
            llm_model_name="gpt-4o",
            llm_temperature=1,
            llm_base_url="",
            llm_api_key="your-api-key",  # Replace with your API key
            use_own_browser=False,
            keep_browser_open=False,
            headless=False,
            disable_security=True,
            window_w=1280,
            window_h=1100,
            save_recording_path="./tmp/record_videos",
            save_agent_history_path="./tmp/agent_history",
            save_trace_path="./tmp/traces",
            enable_recording=True,
            task="Perform a web search for 'OpenAI'",
            add_infos="Additional details",
            max_steps=100,
            use_vision=True,
            max_actions_per_step=10,
            tool_calling_method="auto",
            api_name="/run_with_stream"
        )
        st.write("Gradio API Result:", result)
    except Exception as e:
        st.error(f"Failed to communicate with Gradio: {e}")
else:
    st.error("Gradio server is not running.")

