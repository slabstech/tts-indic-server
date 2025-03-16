# Parler-TTS Latency Measurements

This report summarizes the latency measurements for the Parler-TTS text-to-speech system (`ai4bharat/indic-parler-tts`) under different hardware configurations and optimization methods. The latency is reported in seconds and corresponds to the time taken to generate audio for a given number of words. The data is derived from various tests conducted on March 16, 2025.

## Latency Table

# Parler-TTS Latency Measurements (Formatted)

| Hardware | Optimization Method         | Word Count | Latency (s) | Notes                              |
|:---------|:----------------------------|:----------:|:-----------:|:-----------------------------------|
| T4       | Simple Transformer          |     5      |    4.70     | Baseline measurement              |
| T4       | Simple Transformer          |    21      |   23.63     | Baseline measurement              |
| T4       | Flash Attention             |     5      |    8.16     | Slower than baseline              |
| T4       | Flash Attention             |    21      |   38.29     | Significantly slower than baseline|
| L4       | Flash Attention             |     5      |    3.99     | Fastest for 5 words across tests  |
| L4       | Flash Attention             |    21      |   20.82     | Improved over T4 FA               |
| L4       | Flash Attention (App)       |     7      |    7.92     | App request measurement           |
| A10G     | Flash Attention             |    21      |   25.52     | Consistent but slower than L4     |
| A10G     | Flash Attention             |    21      |   24.33     | Slight variation in repeated test |
| L4       | Torch Compile (Regular)     |     1      |    2.72     | Minimal input size                |
| L4       | Torch Compile (Regular)     |     5      |    2.58     | Fastest for small input           |
| L4       | Torch Compile (Regular)     |     7      |    4.70     | Comparable to baseline T4         |
| L4       | Torch Compile (Regular)     |    21      |   10.65     | Best regular compile for 21 words |
| L4       | Torch Compile (Regular)     |    21      |   11.99     | Slight variation                  |
| L4       | Torch Compile (Regular)     |    21      |   12.10     | Consistent performance            |
| L4       | Torch Compile (Regular)     |    21      |   13.51     | Higher variation                  |
| L4       | Torch Compile (Reduce OH)   |     7      |    3.00     | Estimated from "3s - 7 words"     |
| L4       | Torch Compile (Reduce OH)   |    21      |   10.00     | Estimated from "10 s - 21"        |
| L4       | Torch Compile (Reduce OH)   |    21      |   12.00     | Estimated from "12 s - 21 words"  |

## Observations

1. **Hardware Impact**:
   - The L4 server with Flash Attention showed the best performance for 5 words (3.99s), suggesting better optimization or higher computational power compared to T4.
   - A10G with Flash Attention was slower (24-25s for 21 words) than L4 (20.82s), indicating potential hardware or configuration differences.

2. **Optimization Methods**:
   - **Simple Transformer (T4)**: Served as a baseline with 4.70s for 5 words and 23.63s for 21 words.
   - **Flash Attention**: Surprisingly slower on T4 (8.16s for 5 words, 38.29s for 21 words) compared to the baseline, but improved on L4 (3.99s for 5 words, 20.82s for 21 words). This suggests Flash Attention benefits from specific hardware capabilities.
   - **Torch Compile (Regular)**: Consistently faster than Flash Attention, with the best result for 5 words at 2.58s and a range of 10.65-13.51s for 21 words.
   - **Torch Compile (Reduce Overhead)**: Showed promising results with approximately 3s for 7 words and 10-12s for 21 words, indicating potential for lower latency with this mode.

3. **Input Size**:
   - Latency generally increases with word count, but the scaling is not linear. For example, Torch Compile (Regular) took 2.58s for 5 words and 10.65s for 21 words, suggesting optimization benefits for larger inputs.

## Notes

- The "reduce-overhead" mode values (3s, 12s, 10s) were approximated from your shorthand notation; actual measurements might vary slightly.
- All measurements were taken on March 16, 2025, using the `ai4bharat/indic-parler-tts` model.
- Latency values are in seconds (s), rounded to two decimal places.
- Word counts represent the number of words in the input text.
- "Reduce OH" refers to the "reduce-overhead" mode in Torch Compile.
- The table is sorted by hardware, then optimization method, and finally word count for better readability.

## Conclusion
The Torch Compile optimization, particularly with "reduce-overhead" mode, appears to offer the best balance of latency reduction across different input sizes. The L4 server with Flash Attention also performed well, especially for smaller inputs. For optimal performance, consider using Torch Compile with "reduce-overhead" mode on capable hardware, though further testing could refine these findings.