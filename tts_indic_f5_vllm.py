import torch
import numpy as np
import soundfile as sf
from vllm import LLM

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load IndicF5 with vLLM's Transformers backend
repo_id = "ai4bharat/IndicF5"
try:
    llm = LLM(
        model=repo_id,
        task="generate",  # TTS may not align with "generate"; adjust if needed
        trust_remote_code=True,
        model_impl="transformers",  # Force Transformers backend
        device=device
    )
except Exception as e:
    print(f"Failed to load IndicF5 with vLLM: {e}")
    print("Falling back to Hugging Face Transformers...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device)
    llm = None

# Input parameters
text = "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ, ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ"
ref_audio_path = "prompts/KAN_F_HAPPY_00001.wav"
ref_text = "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."

# Generate speech
if llm:
    try:
        # Attempt vLLM inference
        with torch.no_grad():
            audio = llm.generate(
                text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text
            )
    except Exception as e:
        print(f"vLLM inference failed: {e}")
        print("Using Transformers fallback...")
        llm = None

if not llm:
    # Fallback to original Transformers inference
    model.eval()
    with torch.no_grad():
        audio = model(
            text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text
        )

# Ensure audio is a numpy array and normalize
audio = np.array(audio, dtype=np.float32)
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.max() > 1.0 or audio.min() < -1.0:
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]

# Save output
sf.write("namaste.wav", audio, samplerate=24000)
print("Audio saved as namaste.wav")