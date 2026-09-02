# SerenityPC

SerenityPC is a local-first, privacy-respecting intelligent desktop interface and multi-agent orchestration engine. It runs open-weight large language models directly on consumer hardware with zero cloud telemetry, full offline functionality, multimodal ingestion, and hardware-accelerated local inference.

---

## Quick Setup Guide

### Prerequisites
- **OS**: Windows 10/11 (64-bit).
- **Python**: Python 3.14 (3.10+ compatible).
- **CUDA Toolkit**: NVIDIA CUDA 12.6+ / 13.x with `nvcc` (`sm_75`+ architecture).
- **Build Tools**: Microsoft Visual C++ Build Tools (MSVC 2019/2022/v142+) for native CUDA engine builds.
- **Hardware**: 4-core CPU, 8GB RAM minimum; 6GB+ VRAM NVIDIA GPU, 16–32GB RAM, and SSD storage recommended.

### 1. Installation
Run the self-correcting installer to create an isolated environment and build hardware-optimized inference backends:
```bash
python SETUPfile.py
```
*(Developers can use `python setup.py` for manual compilation, CUDA toolchain isolation, and component inspection).*

### 2. Desktop Shortcut
Generate a clean desktop launcher:
```bash
python System/shortcuts.py
```

### 3. Launch
Launch via the created desktop shortcut, `run.bat`, or command line:
```bash
python main.py
```

---

## Basic Use

1. **Select Models**: Open **Settings** (top-left gear icon) to map GGUF models to desired persona levels.
2. **Choose Persona**: Slide the **Persona** trackbar to the target level (Levels 1–5, or secret Levels 6–7).
3. **Chat**: Enter prompt in the input box and press `Enter` or click `Send`.
4. **Load / Offload**:
   - Click `Begin!` (or click persona level) to load model weights into memory.
   - Button transforms into `Offload` once active; click `Offload` to immediately release VRAM/RAM.
5. **Halt**: Click `Halt` to abort generation mid-stream.

---

## Basic Features & Usage

### 1. Persona Levels
Serenity uses a tiered persona matrix to tailor system prompts, tool capabilities, and reasoning depth:

| Level | Name | Role & Capabilities |
| :--- | :--- | :--- |
| **Lvl 1** | Spark | Rapid responses, short latency, simple tasks. |
| **Lvl 2** | Serenity | General assistant; automated web search and basic tool calling. |
| **Lvl 3** | Forge | Collaborative, structured, project- and code-focused. |
| **Lvl 4** | Empathy | Deep emotional reasoning, interpersonal nuances, reflective dialogue. |
| **Lvl 5** | Sage | Complex academic logic, math, multi-step problem solving, deep context. |
| **Lvl 6** | Transcendent | Secret supervisor tier; dynamic subagent orchestration and tool routing. |
| **Lvl 7** | Cecilia | Master synthesis engine; multi-agent hive mind or direct omniscience. |

*Tip: Click the 'Persona:' header 6 times or double-click the secret anchor to reveal Level 7.*

### 2. Deep Cook (Iterative Reasoning)
Multi-round recursive thinking for complex prompts. Weighs intermediate conclusions against the initial query across configurable cycles (2–12 passes).
- **One-Shot Mode**: Dedicated button loads Deep Cook model for a single intensive query.
- **Toggle Mode**: Toggle switches standard inference into continuous Deep Cook reasoning.

### 3. Multimodal & Media Attachments
- **Attachment (`+`)**: Attach images, audio (.wav, .mp3, .flac, .ogg, .m4a), or documents directly into prompt.
- **Video & Vision**: Process frames via dedicated vision models or native encoder/decoder models.
- **Projector Mapping**: Pair `.mmproj` projector files with models in Settings > Multimedia Handling.

### 4. Reasoning & Agentic Stream Demuxing
- **Thinking Dropdown**: Collapsible `[+] View Thinking Process` isolates model internal chain-of-thought from final user prose.
- **Agentic Actions Dropdown**: Collapsible `[+] View Agentic Actions` traces live subagent delegation and tool executions.

### 5. Pulse / Default Mode Network (DMN)
- Simulates background cognitive consolidation when idle.
- Ponders recent interactions and writes long-term memories to `prime.chronicles.txt`.

### 6. Hardware Tuning & Inference Controls
- **GPU Layer Offload**: Controls layer count pushed to VRAM (`-1` = all layers in VRAM).
- **KV Cache Quantization**:
  - `Q4_0`: 25% base footprint (recommended for tight VRAM).
  - `Q8_0`: 50% footprint (balanced).
  - `FP16`: 100% full precision.
- **Context Size**: Configurable per level (4k, 8k, 16k, 32k, 64k, up to 256k).
- **Sampling**: Temperature, Min-P, Top-P, Top-K, Repetition Penalty, and Batch token size.

### 7. Backend Telemetry & Diagnostic Tabs
- **Log 1 (Thoughts/Status)**: Model initialization, memory allocation, load milestones.
- **Log 2 (Tools)**: Tool executions, web search queries, raw search filtering.
- **Log 3 (Engine)**: Full engine output trace and generation warnings.
- **Log 4 (Diagnostics/Stats)**: Live decode tokens per second (t/s), active context count, VRAM/RAM gauges.

---

## Hardware Requirements & Model Tiers

### System Requirements
- **Minimum**: 4-core CPU, 8GB RAM, 4GB VRAM, 10GB storage.
- **Recommended**: 8-core CPU, 16–32GB RAM, 6–12GB+ VRAM, SSD storage.

### Hardware Tier Matrix
| Tier | VRAM (Min/Rec/Opt) | RAM (Min/Rec/Opt) | Suggested Models |
| :--- | :--- | :--- | :--- |
| **Compact** | 4 / 4 / 6 GB | 8 / 12 / 16 GB | Gemma-4 E2B, Gemma-4 E4B (Q4_K) |
| **Small** | 4 / 6 / 6 GB | 16 / 24 / 32 GB | Gemma-4 E4B, Gemma-4 26B-A4B (MXFP4) |
| **Medium** | 4 / 6 / 8 GB | 20 / 32 / 48 GB | Gemma-4 26B-A4B, Gemma-4 31B (Q4_K) |
| **Performance** | 6 / 8 / 12+ GB | 32 / 48 / 48+ GB | Gemma-4 26B, Gemma-4 31B, Full Context |

### Reference Model Footprints (Q4_0 KV Cache)
| Model | Variant | Weights | KV Cache (4k) | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma-4 E2B** | IQ4 | ~2.76 GiB | ~56.5 MiB | ~2.82 GiB |
| **Gemma-4 E4B** | Q4_K | ~4.74 GiB | ~152.4 MiB | ~4.89 GiB |
| **Gemma-4 26B-A4B** | MXFP4 | ~15.47 GiB | ~250.3 MiB | ~15.72 GiB |
| **Gemma-4 26B-A4B** | Q5_K | ~19.75 GiB | ~250.3 MiB | ~20.00 GiB |
| **Gemma-4 31B** | Q4_K | ~17.46 GiB | ~1,001.2 MiB | ~18.46 GiB |

---

## Profiles, Vault & Theming
- **User Profiles**: Isolated workspaces (`Users/<user>/config.json`).
- **Cryptographic Vault**: AES-256 encrypted chat history with Master Password auto-lock.
- **Theming & Display**: OLED Dark Mode, Frosted Glass, Dynamic Persona palettes, and customizable UI/log font scaling.

---

## Credits & License

### Author & Maintainer
- **GhostHeartZer0**.

### Core Engines & Runtimes
- **Inference Backends**: `llama.cpp` (Georgi Gerganov & contributors) and `llama-cpp-python` (Andrei Betlen & contributors) (MIT).
- **Model Architectures**: Google Gemma & T5Gemma (Google Terms of Use), Qwen (Alibaba Cloud), Granite (IBM Research).
- **Algorithmic Research**: TurboQuant (Google DeepMind), TriAttention (MIT, NVIDIA, Zhejiang University).
- **Supporting Libraries**: PyTorch, OpenCV, Pillow, Sentence-Transformers, ChromaDB, Hugging Face Hub, ONNX Runtime.
- **Hardware Integrations**: MSI Mystic Light SDK (Micro-Star International Co., Ltd.).

### License
- **SerenityPC**: Released under the MIT License (LICENSE.md). Copyright (c) 2026 GhostHeartZer0.