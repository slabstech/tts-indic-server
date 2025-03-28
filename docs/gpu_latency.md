Text = ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ, ಸಾಮ್ರಾಜ್ಯದಲ್ಲಿ ಅತ್ಯುನ್ನತಸ್ಥಾನವನ್ನು ಗಳಿಸಿದೆ. ಇದರ ಕರ್ತೃವಿಗೆ ಆದಿಕವಿ ಆ ಎಂದೂ ಸಾರ್ಧಕವಾದ ವಿಶೇಷಣಗಳು ಬರಾಲಾ ಪ್ರಸಿದ್ಧಿಗೆ ಇಲ್ಲಿನ ಬಂದಿವೆ. ಶ್ರೇಯಸ್ಸಿನ ಸಂಪಾದನೆಗೂ ನಂಬಿಕೆಯಾಗಿದೆ. ಸೀತಾ, ಕಧಾವಸ್ತುವು ರಸಜ್ಞ್ಞ ತೆಯನ್ನು ಸಹೃದಯನ  ಭಾರತದ ಸಹಾಚು ಇವದ್ಮಡಿಸುವುದಲ್ಲದೆ, ಜನರೆಲ್ಲರ ಆತನ ನ್ಯಾಯವಾದ್‌ ವಾಲ್ಮೀಕಿಮಹರ್ಷಿಗಳಂ ಈ ಮಹಾಕಾವ್ಯದಲ್ಲಿ ಪ್ರಧಾನಪಾತ್ರಗಳಾದ


| GPU          | Time Taken | 
|--------------|------------|
| GTX 1060     | N/A        |
| RTX 4050     | N/A        |
| Zero GPU     | 52 s       | 
| T4           | N/A        | 
| L4           | 47 s , 48s |
| L40          | 42 s, 42 s |
| A10G large   | 42 s, 42 s |

025-03-24 09:13:22,578 - tts_indic_server - INFO - Took 58.73 seconds to generate audio for 5 words using CUDA:0
INFO:     10.20.34.20:35651 - "POST /v1/audio/speech HTTP/1.1" 200 OK
INFO:     10.20.28.164:8543 - "GET /v1/audio/speech HTTP/1.1" 405 Method Not Allowed
2025-03-24 09:16:11,712 - tts_indic_server - INFO - Took 2.49 seconds to generate audio for 5 words using CUDA:0
INFO:     10.20.28.164:36826 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-24 09:16:36,489 - tts_indic_server - INFO - Took 2.43 seconds to generate audio for 4 words using CUDA:0
INFO:     10.20.28.164:54388 - "POST /v1/audio/speech HTTP/1.1" 200 OK
W0324 09:18:37.443000 1 torch/_dynamo/convert_frame.py:906] [0/8] torch._dynamo hit config.cache_size_limit (8)
W0324 09:18:37.443000 1 torch/_dynamo/convert_frame.py:906] [0/8]    function: 'forward' (/usr/local/lib/python3.10/dist-packages/parler_tts/modeling_parler_tts.py:2695)
W0324 09:18:37.443000 1 torch/_dynamo/convert_frame.py:906] [0/8]    last reason: 0/0: tensor 'L['cache_position']' size mismatch at index 0. expected 51, actual 1
W0324 09:18:37.443000 1 torch/_dynamo/convert_frame.py:906] [0/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
W0324 09:18:37.443000 1 torch/_dynamo/convert_frame.py:906] [0/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
2025-03-24 09:19:23,662 - tts_indic_server - INFO - Took 80.66 seconds to generate audio for 37 words using CUDA:0
INFO:     10.20.34.20:41797 - "POST /v1/audio/speech HTTP/1.1" 200 OK
2025-03-24 09:26:25,557 - tts_indic_server - INFO - Took 46.08 seconds to generate audio for 37 words using CUDA:0
INFO:     10.20.34.20:56778 - "POST /v1/audio/speech HTTP/1.1" 200 OK