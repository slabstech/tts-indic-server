import gradio as gr
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to send text input to the API and retrieve the audio file
def get_audio(input_text, voice_description):
    try:
        # Define the API endpoint and headers
        url = "http://localhost:9860/v1/audio/speech"  # Note: Added http://
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Define the request payload
        payload = {
            "input": input_text,
            "voice": voice_description
        }
        
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"API request successful. Status code: {response.status_code}")
            
            # Save the audio file
            audio_file_path = "output_audio.mp3"
            with open(audio_file_path, "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_file.write(chunk)
            
            logger.info(f"Audio file saved to: {audio_file_path}")
            
            # Return the path to the saved audio file
            return audio_file_path
        else:
            logger.error(f"API request failed. Status code: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return f"Request error: {e}"
    except Exception as e:
        logger.error(f"General exception: {e}")
        return f"Error: {e}"

# Predefined voice descriptions for dropdown
voice_descriptions = [
    "A male speaker delivers high-pitched, engaging voice is captured in a clear, close-sounding recording. His slightly slower delivery conveys a positive tone.",
    "A female speaker speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality.",
    "A female speaker speaks with a slightly higher pitch in a close-sounding environment. Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording."
]

# Define the Gradio interface
demo = gr.Interface(
    fn=get_audio,
    inputs=[
        gr.Textbox(label="Enter Text", placeholder="Type your text here..."),
        gr.Dropdown(choices=voice_descriptions, label="Select Voice Description")
    ],
    outputs=gr.Audio(label="Generated Audio"),
)

# Launch the Gradio demo
try:
    demo.launch()
except Exception as e:
    logger.error(f"Failed to launch Gradio demo: {e}")
