# TTS Indic Server

## Overview
Text to Speech (TTS) for Indian languages using [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)  model.

## Table of Contents
- [Live Server](#live-server)
- [Usage](#usage)
  - [How to Use the Service](#how-to-use-the-service)
    - [High Latency Service](#high-latency-service)
    - [Low Latency Service](#low-latency-service)
- [Getting Started - Development](#getting-started---development)
  - [For Production (Docker)](#for-production-docker)
  - [For Development (Local)](#for-development-local)
- [Downloading Indic TTS Model](#downloading-indic-tts-model)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
  - [Examples](#examples)
    - [Kannada](#kannada)
    - [Hindi](#hindi)
  - [Specifying a Different Format](#specifying-a-different-format)
  - [Playing Back the Audio](#playing-back-the-audio)
  - [Describing the Voice](#describing-the-voice)
    - [Describing the Voice with Specific Speaker - Suresh](#describing-the-voice-with-specific-speaker---suresh)
    - [Describing the Voice with Specific Speaker - Anu](#describing-the-voice-with-specific-speaker---anu)
- [Building Docker Image](#building-docker-image)
- [Run the Docker Image](#run-the-docker-image)
- [Available Speakers](#available-speakers)
- [Tips](#tips)
- [Description Examples](#description-examples)
- [Citations](#citations)

## Live Server

We have hosted a Text to Speech (TTS) service that can be used to verify the accuracy of Speech generation. The service is available in two modes:


### Low Latency, Fast System (Available on Request)
- **URL**: [Low Latency TTS Service](https://huggingface.co/spaces/gaganyatri/tts_indic_server)

<!--
## Usage

### How to Use the Service

You can test the service using `curl` commands. Below are examples for both service modes:

#### Paused, On-demand, $.05 /hour, Fast

```bash kannada_example.sh
curl -X 'POST' \
  'https://gaganyatri-tts-indic-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.", "voice": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speakers voice sounding clear and very close up."}'  -o audio_kannada_gpu_cloud.mp3
```
-->

## Getting Started - Development

### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```
  For CPU only
  ```bash
  docker compose -f cpu-compose.yaml up -d
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
  - For GPU
      ```bash
      pip install -r requirements.txt
      ```
  - For CPU only
      ```bash
      pip install -r cpu-requirements.txt
      ```

## Downloading Indic TTS Model

```bash download_model.sh
huggingface_cli download ai4bharat/indic-parler-tts
```

## Running with FastAPI Server
Run the server using FastAPI with the desired language (e.g., Kannada):
- for GPU
  ```bash
  python src/tts_api.py --port 7860 --host 0.0.0.0 --device gpu
  ```
- for CPU only
  ```bash
  python src/tts_api.py --port 7860 --host 0.0.0.0 --device cpu
  ```

## Evaluating Results
You can evaluate the ASR transcription results using `curl` commands. Below are examples for Kannada audio samples.

#### Kannada

```bash kannada_example.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ."}' -o audio_kannada.mp3
```

#### Hindi

```bash hindi_example.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "अरे, तुम आज कैसे हो?"}' -o audio_hindi.mp3
```

### Specifying a Different Format

```bash specify_format.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "Hey, how are you?", "response_type": "wav"}' -o audio.wav
```

### Describing the Voice

```bash audio_kannada_describe_voice.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.", "voice": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speakers voice sounding clear and very close up."}'  -o audio_kannada_describe_voice.mp3
```

#### Describing the Voice with Specific Speaker - Suresh

```bash audio_kannada_describe_voice_specific_speaker_suresh.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.", "voice": "Suresh  speaks at a moderate pace with a slightly monotone tone. The recording is clear, with a close sound and only minimal ambient noise."}'  -o audio_kannada_describe_voice_specfic_speaker_suresh.mp3
```

#### Describing the Voice with Specific Speaker - Anu

```bash audio_kannada_describe_voice_specific_speaker_anu.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.", "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality."}'  -o audio_kannada_describe_voice_specfic_speaker_anu.mp3
```



## Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/tts_indic_server -f Dockerfile .
```

### Run the Docker Image
```bash
docker run --gpus all -it --rm -p 7860:7860 slabstech/tts_indic_server
```

## Available Speakers

The model includes **69 speakers** across 18 officially supported languages, with each language having a set of recommended voices for optimal performance. Below is a table summarizing the available speakers for each language, along with the recommended ones.

| **Language**      | **Available Speakers**                                       | **Recommended Speakers**       |
|-------------------|-------------------------------------------------------------|---------------------------------|
| Assamese          | Amit, Sita, Poonam, Rakesh                                  | Amit, Sita                      |
| Bengali           | Arjun, Aditi, Tapan, Rashmi, Arnav, Riya                    | Arjun, Aditi                    |
| Bodo              | Bikram, Maya, Kalpana                                       | Bikram, Maya                    |
| Chhattisgarhi     | Bhanu, Champa                                              | Bhanu, Champa                   |
| Dogri             | Karan                                                      | Karan                           |
| English           | Thoma, Mary, Swapna, Dinesh, Meera, Jatin, Aakash, Sneha, Kabir, Tisha, Chingkhei, Thoiba, Priya, Tarun, Gauri, Nisha, Raghav, Kavya, Ravi, Vikas, Riya | Thoma, Mary                     |
| Gujarati          | Yash, Neha                                                  | Yash, Neha                      |
| Hindi             | Rohit, Divya, Aman, Rani                                   | Rohit, Divya                    |
| Kannada           | Suresh, Anu, Chetan, Vidya                                 | Suresh, Anu                     |
| Malayalam         | Anjali, Anju, Harish                                       | Anjali, Harish                  |
| Manipuri          | Laishram, Ranjit                                           | Laishram, Ranjit                |
| Marathi           | Sanjay, Sunita, Nikhil, Radha, Varun, Isha                  | Sanjay, Sunita                  |
| Nepali            | Amrita                                                     | Amrita                          |
| Odia              | Manas, Debjani                                             | Manas, Debjani                  |
| Punjabi           | Divjot, Gurpreet                                           | Divjot, Gurpreet                |
| Sanskrit          | Aryan                                                      | Aryan                           |
| Tamil             | Kavitha, Jaya                                              | Jaya                            |
| Telugu            | Prakash, Lalitha, Kiran                                    | Prakash, Lalitha                |

## Tips
* We've set up an [inference guide](https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md) to make generation faster. Think SDPA, torch.compile, batching and streaming!
* Include the term "very clear audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
* Punctuation can be used to control the prosody of the generations, e.g., use commas to add small breaks in speech
* The remaining speech features (gender, speaking rate, pitch, and reverberation) can be controlled directly through the prompt

## Description Examples

1. **Aditi - Slightly High-Pitched, Expressive Tone**:
   _"Aditi speaks with a slightly higher pitch in a close-sounding environment. Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording."_

2. **Sita - Rapid, Slightly Monotone**:
   _"Sita speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment with excellent recording quality."_

3. **Tapan - Male, Moderate Pace, Slightly Monotone**:
   _"Tapan speaks at a moderate pace with a slightly monotone tone. The recording is clear, with a close sound and only minimal ambient noise."_

4. **Sunita - High-Pitched, Happy Tone**:
   _"Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, and the recording is of excellent quality."_

5. **Karan - High-Pitched, Positive Tone**:
   _"Karan’s high-pitched, engaging voice is captured in a clear, close-sounding recording. His slightly slower delivery conveys a positive tone."_

6. **Amrita - High-Pitched, Flat Tone**:
   _"Amrita speaks with a high pitch at a slow pace. Her voice is clear, with excellent recording quality and only moderate background noise."_

7. **Aditi - Slow, Slightly Expressive**:
   _"Aditi speaks slowly with a high pitch and expressive tone. The recording is clear, showcasing her energetic and emotive voice."_

8. **Young Male Speaker, American Accent**:
   _"A young male speaker with a high-pitched American accent delivers speech at a slightly fast pace in a clear, close-sounding recording."_

9. **Bikram - High-Pitched, Urgent Tone**:
   _"Bikram speaks with a higher pitch and fast pace, conveying urgency. The recording is clear and intimate, with great emotional depth."_

10. **Anjali - High-Pitched, Neutral Tone**:
    _"Anjali speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality."_

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate

- Reference
    - [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)

## Citations

```bibtex citation_1.bib
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```bibtex citation_2.bib
@misc{lyth2024natural,
  title = {Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
  author = {Dan Lyth and Simon King},
  year = {2024},
  eprint = {2402.01912},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD}
}
```