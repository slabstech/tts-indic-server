

curl -X POST "http://localhost:8000/synthesize" \
-H "Content-Type: application/json" \
-d '{"text": "ನಾನು ಯಾವುದೇ ಚಿಂತೆ ಇಲ್ಲದೆ ನನ್ನ ಸ್ನೇಹಿತರನ್ನು ನನ್ನ ಆಟೋಮೊಬೈಲ್ ತಜ್ಞರ ಬಳಿಗೆ ಕಳುಹಿಸುತ್ತೇನೆ.", "ref_audio_name": "KAN_F (Happy)", "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."}' \
-o output.wav



curl -X POST "https://gaganyatri-tts-indic-f5.hf.space/synthesize" \
-H "Content-Type: application/json" \
-d '{"text": "ನಾನು ಯಾವುದೇ ಚಿಂತೆ ಇಲ್ಲದೆ ನನ್ನ ಸ್ನೇಹಿತರನ್ನು ನನ್ನ ಆಟೋಮೊಬೈಲ್ ತಜ್ಞರ ಬಳಿಗೆ ಕಳುಹಿಸುತ್ತೇನೆ.", "ref_audio_name": "KAN_F (Happy)", "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."}' \
-o output.wav

