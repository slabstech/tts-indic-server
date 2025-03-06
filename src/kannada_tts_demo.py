import gradio as gr
import requests

# Function to call the API and generate audio
def generate_audio(input_text):
    # API Endpoint and Headers
    url = "https://gaganyatri-tts-indic-server-cpu.hf.space/v1/audio/speech"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Payload with input text and voice configuration
    payload = {
        "input": input_text,
        "voice": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speakers voice sounding clear and very close up."
    }
    
    # Sending POST request to the API
    response = requests.post(url, json=payload, headers=headers)
    
    # Check for successful response
    if response.status_code == 200:
        # Save audio file locally
        audio_file = "output_audio.mp3"
        with open(audio_file, "wb") as f:
            f.write(response.content)
        return audio_file  # Return the file path for Gradio's Audio component
    else:
        # Handle errors
        return f"Error: {response.status_code}, {response.text}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Kannada Text-to-Speech Generator")
    gr.Markdown(
        "Enter Kannada text below, and the application will generate an audio file using the specified voice."
    )
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter Kannada Text",
            placeholder="Type your text here...",
            lines=3,
        )
    
    with gr.Row():
        submit_button = gr.Button("Generate Audio")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio", type="filepath")
    
    # Link input, function, and output
    submit_button.click(generate_audio, inputs=text_input, outputs=audio_output)

# Launch the Gradio app
demo.launch()
