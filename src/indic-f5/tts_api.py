import io
import torch
import librosa
import requests
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModel
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="IndicF5 TTS API", description="Text-to-Speech API for Indian Languages")

# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)

# Supported languages
SUPPORTED_LANGUAGES = [
    "assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "odia", "punjabi", "tamil", "telugu"
]

# Function to load audio from URL or uploaded file
def load_audio(audio_content: bytes):
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_content))
        return sample_rate, audio_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {str(e)}")

# Core synthesis function
def synthesize_speech(text: str, ref_audio: bytes, ref_text: str):
    if not ref_audio or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference audio and text are required.")

    # Load reference audio
    sample_rate, audio_data = load_audio(ref_audio)

    # Save reference audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

        # Generate speech
        audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    # Normalize output
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Save generated audio to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, audio, 24000, format='WAV')
    output_buffer.seek(0)

    return output_buffer

# Pydantic model for request validation
class TTSRequest(BaseModel):
    text: str
    ref_text: str

# Default endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to IndicF5 TTS API. Use /synthesize/{language} to generate speech."}

# Dynamic endpoint for each language
for language in SUPPORTED_LANGUAGES:
    @app.post(f"/synthesize/{language}", response_class=StreamingResponse)
    async def synthesize_language(
        text: str = Form(...),
        ref_text: str = Form(...),
        ref_audio: UploadFile = File(...)
    ):
        try:
            # Read uploaded audio file
            ref_audio_content = await ref_audio.read()

            # Synthesize speech
            audio_buffer = synthesize_speech(text, ref_audio_content, ref_text)

            # Return audio as streaming response
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename={language}_output.wav"}
            )
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device), "supported_languages": SUPPORTED_LANGUAGES}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)