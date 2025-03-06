
import gradio as gr
import requests
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the JSON file containing use cases
with open("voice_description_indian.json", "r") as file:
    usecases = json.load(file)

# Function to send text input to the API and retrieve the audio file
def get_audio(input_text, usecase_id):
    try:
        # Find the corresponding use case by ID
        usecase = next((uc for uc in usecases["usecases"] if uc["id"] == usecase_id), None)
        if not usecase:
            return f"Error: Use case with ID {usecase_id} not found."

        voice_description = usecase["voice_description"]
        print(voice_description)
        print(input_text)
        # Define the API endpoint and headers
        url = "https://gaganyatri-indic-all-server.hf.space/v1/audio/speech"
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
            audio_file_path = usecase["output_filename"]
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

# Define Gradio interface inputs and outputs
demo = gr.Interface(
    fn=get_audio,
    inputs=[
        gr.Textbox(label="Enter Text", placeholder="Type your text here..."),
        gr.Dropdown(
            label="Select Use Case",
            choices=[f"{uc['id']}: {uc['voice_description']}" for uc in usecases["usecases"]],
            type="index"
        )
    ],
    outputs=gr.Audio(label="Generated Audio"),
)

# Launch the Gradio demo
try:
    demo.launch()
except Exception as e:
    logger.error(f"Failed to launch Gradio demo: {e}")
