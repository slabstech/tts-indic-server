# TTS Indic Server

## Overview
Automatic Speech Recognition (ASR) for Indian languages using IndicConformer models. The default model is set to Kannada ASR.


## Demo Video

Watch a quick demo of our project in action! Click the image below to view the video on YouTube.

<a href="https://youtu.be/F0Mo0zjyysM" target="_blank">
  <img src="https://img.youtube.com/vi/F0Mo0zjyysM/0.jpg" alt="Watch the video">
</a>


## Table of Contents
- [Supported Languages](#supported-languages)
- [Getting Started](#getting-started)
- [For Production (Docker)](#for-production-docker)
- [For Development (Local)](#for-development-local)
- [Downloading Translation Models](#downloading-translation-models)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
- [Kannada Transcription Examples](#kannada-transcription-examples)
- [Building Docker Image](#building-docker-image)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Additional Resources](#additional-resources)

## Supported Languages - 22
| Language      | Code  |
|---------------|-------|
| Assamese      | `as`  |
| Bengali       | `bn`  |
| Bodo          | `brx` |
| Dogri         | `doi` |
| Gujarati      | `gu`  |
| Hindi         | `hi`  |
| Kannada       | `kn`  |
| Kashmiri      | `ks`  |
| Konkani       | `kok` |
| Maithili      | `mai` |
| Malayalam     | `ml`  |
| Manipuri      | `mni` |
| Marathi       | `mr`  |
| Nepali        | `ne`  |
| Odia          | `or`  |
| Punjabi       | `pa`  |
| Sanskrit      | `sa`  |
| Santali       | `sat` |
| Sindhi        | `sd`  |
| Tamil         | `ta`  |
| Telugu        | `te`  |
| Urdu          | `ur`  |

## Getting Started

### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  ```bash
  docker compose -f compose.yaml up -d
  ```
  2. **Update source and target languages**:
  Modify the `compose.yaml` file to set the desired language. Example configurations:
  - **Kannada**:
  ```yaml
  language: kn
  ```
  - **Hindi**:
  ```yaml
  language: hi
  ```

### For Development (Local)
- **Prerequisites**: Python 3.6+
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## Downloading Translation Models
Models can be downloaded from AI4Bharat's HuggingFace repository:

### Kannada
```bash
huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large
```

### Other Languages

#### Malayalam
```bash
huggingface-cli download ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large
```

#### Hindi
```bash
huggingface-cli download ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large
```

## Running with FastAPI Server
Run the server using FastAPI with the desired language (e.g., Kannada):
```bash
python src/asr_indic_server/asr_api.py --port 8000 --language kn --host 0.0.0.0
```


## Evaluating Results
You can evaluate the ASR transcription results using `curl` commands. Below are examples for Kannada audio samples.
**Note**: GitHub doesn’t support audio playback in READMEs. Download the sample audio files and test them locally with the provided `curl` commands to verify transcription results.

### Kannada Transcription Examples

#### Sample 1: kannada_sample_1.wav
- **Audio File**: [samples/kannada_sample_1.wav](samples/kannada_sample_1.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:8000/transcribe/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_1.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು```
Translation: "What is the capital of Karnataka"

#### Sample 2: kannada_sample_2.wav
- **Audio File**: [samples/kannada_sample_2.wav](samples/kannada_sample_2.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:8000/transcribe/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_2.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ```

#### Sample 3 - Song - 4 minutes
- [YT Video- Navaduva Nudiye](https://www.youtube.com/watch?v=LuZzhMN8ndQ)
- **Audio File**: [samples/kannada_sample_3.wav](samples/kannada_sample_3.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:8000/transcribe/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_3.wav;type=audio/x-wav'
```
- **Expected Output**: [kannada_sample_3_out.md](docs/kannada_sample_3_out.md)

#### Sample 4 - Song - 6.4 minutes
- [YT Video- Aagadu Yendu](https://www.youtube.com/watch?v=-Oryie1c-gs)
- **Audio File**: [samples/kannada_sample_4.wav](samples/kannada_sample_4.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:8000/transcribe/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_4.wav;type=audio/x-wav'
```
- **Expected Output**: [kannada_sample_4_out.md](docs/kannada_sample_4_out.md)

**Note**: The ASR does not provide sentence breaks or punctuation (e.g., question marks). We plan to integrate an LLM parser for improved context in future updates.

## Batch Transcription Examples

### Transcribe Batch Endpoint
The `/transcribe_batch` endpoint allows you to transcribe multiple audio files in a single request. This is useful for batch processing of audio files.

- **Command**:
```bash
curl -X 'POST' \
'http://localhost:8000/transcribe_batch/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'files=@samples/kannada_sample_1.wav;type=audio/x-wav' \
-F 'files=@samples/kannada_sample_2.wav;type=audio/x-wav'
```
- **Expected Output**:
```json
{
  "transcriptions": [
    "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು",
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ"
  ]
}
```

## Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/asr_indic_server -f Dockerfile .
```

## Troubleshooting
- **Docker fails to start**: Ensure Docker is running and the `compose.yaml` file is correctly formatted.
- **Transcription errors**: Verify the audio file is in WAV format, mono, and sampled at 16kHz. Adjust using:
```bash
ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
```
- **Model not found**: Download the required models using the `huggingface-cli download` commands above.
- **Port conflicts**: Ensure port 8000 is free when running the FastAPI server.

## References
- [AI4Bharat IndicConformerASR GitHub Repository](https://github.com/AI4Bharat/IndicConformerASR)
- [Nemo - AI4Bharat](https://github.com/AI4Bharat/NeMo)
- [IndicConformer Collection on HuggingFace](https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f)

## Additional Resources

### Running Nemo Model
1. Download the Nemo model:
```bash
wget https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo -O kannada.nemo
```
2. Adjust the audio:
```bash
ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
```
3. Run the program:
```bash
python nemo_asr.py
```

### Running with Transformers
```bash
python hf_asr.py
```

- server-setup.sh - Use for container deployment on OlaKrutrim AI Pod