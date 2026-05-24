# ASR Shootout — Benchmarking Indian Conversational Speech Recognition Systems

## Overview

This project benchmarks multiple Automatic Speech Recognition (ASR) systems on real-world Indian conversational speech collected under noisy and telephony-like conditions.

The evaluation focuses specifically on:

* Bangalore locality name recognition
* Hindi / Hinglish conversational speech
* Noisy real-world audio
* Entity extraction robustness
* Latency vs accuracy tradeoffs

The benchmark compares:

* Open-source ASR models
* Commercial ASR APIs
* Enterprise cloud speech systems

---

# Models Evaluated

| Model                 | Type                              |
| --------------------- | --------------------------------- |
| Whisper Tiny          | Lightweight local ASR             |
| Whisper Base          | Balanced multilingual ASR         |
| Whisper Small	        | Mid-sized multilingual ASR        |
| Whisper Large	        | High-accuracy large-scale         |
| Deepgram Nova-2       | Commercial conversational ASR API |
| Google Speech-to-Text | Enterprise cloud ASR              |

---

# Motivation

Traditional ASR benchmarks often focus only on Word Error Rate (WER).
However, for production hiring platforms, correctly identifying critical entities such as locality names is often more important than perfect sentence transcription.

This benchmark evaluates:

* transcription quality,
* locality entity preservation,
* robustness under noisy conditions,
* inference latency,
* production deployment tradeoffs.

---

# Dataset

## Self-Recorded Conversational Audio

A custom dataset of conversational Bangalore locality recordings was created.

Characteristics:

* Hindi / Hinglish speech
* Natural conversational phrasing
* Mobile phone recordings
* Quiet + traffic + noisy environments
* Real-world speaking speed variations

Examples:

* “Haan main Koramangala side rehta hu”
* “Whitefield tak daily travel karta hu”
* “Silk board ke paas bahut traffic tha”

---

# Project Structure

```text
automatic-speech-recognition/
│
├── data/
│   ├── processed_audio/
|   ├── raw_audio/
│   └── transcripts/
│
├── models/
│   ├── whisper_local.py
│   ├── whisper_tiny.py
|   ├── whisper_small.py
|   ├── whisper_large.py
│   ├── deepgram_api.py
│   └── google_stt.py
│
├── evaluation/
│   ├── metrics.py
│   ├── entity_utils.py
│   └── summary_report.py
│
├── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Evaluation Metrics

## 1. Word Error Rate (WER)

Measures transcription accuracy at word level.

## 2. Character Error Rate (CER)

Measures fine-grained transcription quality.

## 3. Entity Recognition Accuracy

Measures whether the ASR system correctly preserved the Bangalore locality name.

This is the most important production metric for this benchmark.

## 4. Latency

Measures transcription response time.

---

# Benchmark Pipeline

The benchmark pipeline performs:

1. Audio loading
2. Model inference
3. WER/CER computation
4. Entity extraction evaluation
5. Latency measurement
6. Summary generation
7. Plot generation

Entire pipeline runs with:

```bash
python main.py
```

---

# Installation

## Clone Repository

```bash
git clone <repo-url>
cd automatic-speech-recognition
```

---

# Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

## Windows

```bash
.venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# FFmpeg Setup

Whisper requires FFmpeg.

1. Download FFmpeg
2. Add `ffmpeg/bin` to system PATH

Verify:

```bash
ffmpeg -version
```

---

# Google STT Setup

1. Create Google Cloud project
2. Enable Speech-to-Text API
3. Create Service Account
4. Download credentials JSON

Set environment variable:

## Windows CMD

```bash
set GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json
```

## PowerShell

```bash
$env:GOOGLE_APPLICATION_CREDENTIALS="google_credentials.json"
```

---

# Running the Benchmark

```bash
python main.py
```

---

# Output Files

Generated automatically inside:

```text
results/
```

Outputs:

* `benchmark_results.csv`
* `summary_report.csv`
* `wer_comparison.png`
* `latency_comparison.png`
* `entity_accuracy.png`
* `condition_analysis.png`

---

# Key Insights

## Whisper Tiny

* Lowest latency
* Weakest transcription robustness
* Suitable for lightweight edge deployment

## Whisper Base

* Better multilingual robustness
* Higher latency
* Better conversational understanding

## Deepgram

* Strong conversational handling
* Fast response time
* Good production API usability

## Google STT

* Most production-ready cloud ASR
* Strong multilingual handling
* Better robustness under noisy speech

---

# Failure Analysis

Common failure modes observed:

* Locality name distortion
* Hindi-English code-switch confusion
* Noise sensitivity
* Traffic-condition degradation
* Entity substitution despite semantically correct transcription

Example:

Ground Truth:

```text
Thanisandra main shift hone ka plan hai
```

Deepgram Output:

```text
तनी संतरा में shift होने का plan है
```

Sentence meaning remained partially understandable, but locality extraction failed.

This demonstrates why WER alone is insufficient for production evaluation.

---

# Production Considerations

Beyond transcription accuracy, production ASR systems must consider:

* latency,
* streaming support,
* deployment cost,
* multilingual support,
* offline capability,
* infrastructure requirements,
* entity preservation reliability.

---

# Technologies Used

* Python
* OpenAI Whisper
* Deepgram API
* Google Speech-to-Text
* Pandas
* JiWER
* RapidFuzz
* Matplotlib
* Seaborn

---

# Future Improvements

Potential extensions:

* Streaming ASR benchmarking
* Speaker diarization
* Noise augmentation testing
* GPU inference optimization
* Regional language evaluation
* Confidence-score calibration
* Real-time telephony simulation

---

# Conclusion

This benchmark demonstrates that:

* low WER does not guarantee reliable entity extraction,
* conversational Indian ASR remains challenging under noisy conditions,
* deployment constraints significantly affect model choice,
* latency vs robustness tradeoffs are critical in production systems.

The evaluation highlights the importance of task-specific metrics rather than relying solely on generic transcription accuracy.

---

# Author

Shreya Bhat

