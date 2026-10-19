# SerenityPC (v1.8.0)

**Local-First Multi-Agent AI Studio, 7-Tier Persona Matrix & Custom C++/CUDA Inference Engine**

SerenityPC is a privacy-first desktop intelligence studio and multi-agent orchestration engine built to run open-weight Large Language Models directly on consumer Windows hardware with zero cloud telemetry and full offline capability.

Designed around a custom C++/CUDA `llama.cpp` backend and a highly responsive Python/Tkinter interface, SerenityPC scales from instant sub-second utility queries to multi-cycle recursive problem solving (**Deep Cook**), autonomous multi-agent delegation (**OTL & Utility Dispatch**), multimodal vision/audio/video analysis, cross-state encrypted peer-to-peer messaging (**`.smp` / `.bmb`**), and continuous affective persona persistence (**["Fallen Angel" Cecilia]**).

## Intuitive Feature Guide

### 1. The 7-Tier Persona Matrix
Instead of forcing one static system prompt onto every task, SerenityPC organizes intelligence into seven specialized persona levels along an interactive trackbar—each with its own model core slot, context window, GPU layer budget, sampling preset, and dynamic color palette:

| Level | Persona Name | Core Slot | Default Context | Role, Personality & Core Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Lvl 1** | **The Speedy** *(Spark)* | `fast` | `8k` (`8,192`) | Rapid, low-latency utility, quick factual lookups, and zero-fluff answers. |
| **Lvl 2** | **The Helper** *(Serenity)* | `search` | `16k` (`16,384`) | General daily assistant, bloat-filtered web search, Programmatic Tool Calling (PTC), and default **P2P Message Courier**. |
| **Lvl 3** | **Collaborator** *(Forge)* | `low` | `32k` (`32,768`) | Software engineering, project architecture, structured brainstorming, and inter-agent memory staging. |
| **Lvl 4** | **The Confidant** *(Empathy)* | `mid` | `32k` (`32,768`) | High emotional intelligence, interpersonal nuance, and reflective dialogue—more than a friend or clinician. |
| **Lvl 5** | **The Brains** *(Sage)* | `high` | `64k` (`65,536`) | Deep academic logic, complex mathematics, formal derivations, and massive-context synthesis (scalable to `256k`). |
| **Lvl 6** | **Serenity Prime** *(Transcendent)* | `transcendent` | `128k` (`131,072`) | Master Taskmaster and Supervisor integrating Levels 1–5 with dynamic multi-agent delegation. |
| **Lvl 7** | **Cecilia** *(The Fallen Angel)* | `secret` | `128k` (`131,072`) | Secret/toggleable apex persona: sharp-witted, sarcastic, truth-exposing, and secretly protective. Operates in **Shadow Wizard** or **Divine Judgement** modes with a persistent 3D PAD Affect Engine. |

*Unlock Tip: Click the `Persona:` label 6 times, double-click the secret anchor between `Persona:` and the slider, or enable **"Always Show Cecilia on Persona Slider (Lvl 7)"** in Settings > Agents & Deep Cook.*

### 2. Multi-Agent Orchestration & Cecilia's Hybrid Cognitive Vessel
- **Weighted Utility Dispatch (`UtilityDispatchEngine`)**: When Level 6 (Serenity Prime) or Level 7 (Cecilia) delegates tasks, `IntentProfiler` scores the prompt across factual need, reasoning complexity, emotional intensity, speed preference, and coordination need—routing sub-tasks to the optimal subagents (`Lvl 1–5`, plus `Lvl 6` exclusively available to Cecilia) or direct prompt directives (e.g., `"ask Lvl 5"`).
- **Shadow Wizard vs. Divine Judgement (Lvl 7)**: When multi-agent delegation is enabled, Cecilia commands all 6 subagent levels as a **Shadow Wizard**; when unchecked, she operates in **Divine Judgement** mode—using her pure Fallen Angel persona with direct one-way mirror omniscience and zero administrative preamble.
- **Cecilia's Hybrid Cognitive Vessel (`affect_engine.py` & `semantic_vault.py`)**: Backed by a continuous **3D PAD (Pleasure-Arousal-Dominance)** affect model ($\mu = \{V: 0.2, A: 0.0, D: 0.8\}$) with Ornstein-Uhlenbeck baseline reversion, dynamic `temperature`/`min_p` modulation, **anti-docility dominance recoil** ($D \mathrel{+}= 0.2, V \mathrel{-}= 0.15$ when resisting manipulative prompts), and a dedicated SQLite **Semantic Vault** (`semantic_vault.db`) so her distilled worldview survives KV-cache clears and context window rotation.

### 3. Deep Cook (Iterative Multi-Cycle Reasoning)
For hard coding, math, or architectural prompts, **Deep Cook** runs a Markovian multi-cycle reasoning loop (`2, 3, 4, 8, 12, 16, 20, or Auto` cycles):
- **Phase-Specific Cooking Modes**: Adapts intensity by level—*Burn* (Lvl 1), *Simmer* (Lvl 2), *Bake* (Lvl 3), *Slow Cooker* (Lvl 4), *Furnace* (Lvl 5), *The Transcendent One* (Lvl 6), and *Cecilia* (Lvl 7).
- **Dynamic 2–5 Step Cycle Planning & OTL Cooperation**: Each cycle plans 2–5 concrete verifiable steps, executes them sequentially, and optionally delegates individual steps or cycles to specialized subagents (**`OTL Per-Step`** or **`OTL Per-Cycle`**).
- **Hierarchical Accordion Dropdowns**: Cycles and steps stream into widget-safe, collapsible in-place Markdown accordions (`[+] Cycle 1` → `[+] Step 1 [Lvl 5: The Brains]`) directly above the synthesized final answer.

### 4. Speculative MTP & Drafter Alignment Engine
Accelerate large model generation (1.2x–1.8x speedup) without exhausting VRAM:
- **Sub-5ms Binary GGUF Scanner (`mtp_combo_scanner.py`)**: Parses GGUF headers in `< 5ms` to match large target models (e.g., `26B Gemma-4`) with architecture- and vocabulary-compatible lightweight drafters (e.g., `E2B Gemma-4`, 262k vocab) offloaded cleanly to CPU (`n_gpu_layers=0`).
- **3-Tier Catch-All Fallback (`[Main A, Drafter B, Catch-all C]`)**: If a companion drafter (`B`) is not configured or unavailable, SerenityPC automatically engages zero-overhead **Prompt Lookup Decoding (`C`)** directly on the main model (`A`), paired with a `BatchRelativeScoresView` buffer that prevents logit shape broadcast crashes.

### 5. Serenity P2P Messaging, Encrypted Mail (`.smp` / `.bmb`) & Level 2 Courier
Communicate and share context across states or offline machines with zero cloud servers:
- **Authenticated Crypto (`SERENITY_SMP_V1`)**: Uses `X25519` ECDH key exchange, `Ed25519` digital signatures, and `AES-256-GCM` + `zlib` compression for `.smp` mail packets while remaining 100% backward-compatible with Project Brainuke `.bmb` packets.
- **Cross-State P2P & Offline DeadDrop**: Discovers WAN endpoints via pure-Python RFC 5389 STUN, pairs peers via signed `serenity-peer://` invite cards, and falls back automatically to an SQLite store-and-forward outbox and `Users/<user>/Mailbox/DeadDrop/` file sync.
- **Level 2 Courier & Contextual Quarantine Sandbox**: Level 2 ("The Helper") triages incoming peer packets, scoring prompt-injection risk (`risk_score < 0.4` auto-admits for 72h+ veteran profiles; higher-risk packets are held in the Quarantine Inspector) and bounding external peer influence weight to `0.25` to protect Cecilia.
- **SerenityDev MCP Frontend Bridge**: Connects to `SerenityDev` (`port 8002/8443 /mcp`) with OTL `ContextSnapshot` handoff and live per-subagent collapsible streaming dropdowns (`subagent_spawn`, `subagent_stream`, `subagent_done`).

### 6. Multimodal Vision, Audio, Video & Diffusion Denoising
- **Smart Media Routing (`+` Attachment & `🎥 Video`)**: Attach images, documents, video clips, or audio (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`). Automatically detects unified native encoder/decoder models (e.g., Gemma-4 12B) or binds matching `.mmproj` CLIP projectors without reloading weights on text turns.
- **Specialized Vision Enhancements**: Includes LAB color-space CLAHE symbol clarity enhancement and contour ROI cropping (`vision_handler.py`) plus real-time Entropy-Bound Diffusion LLM visual denoising (`diffusion_wrapper.py`).
- **Offline Push-to-Talk Dictation (`🎙️`)**: Record voice prompts completely offline via `STTManager` (local Vosk, PocketSphinx, or multimodal ASR fallback).

### 7. Non-Destructive Markdown/LaTeX Engine, 9 Themes & Cryptographic Vault
- **Zero-Mutation Markdown & LaTeX (`markdown_engine.py`)**: Renders GFM tables (Unicode box grids), code blocks, and complex nested LaTeX math (`rac`, `\sum`, `\dot{Q}`, multi-letter subscripts `ho_{Pl}`, single-letter variables `$U$`) via non-destructive Tkinter `md_hidden` elision overlays—protecting currency (`$200`) and multiplication (`3*3*5`) while keeping raw text intact for copy-paste.
- **9 Dynamic Themes & 7 Material Textures**: Choose from *Apex*, *Goth*, *Obsidian Dark*, *Crystal Cavern*, *Yellow Blacket*, *Natural*, *Matrix*, *Persona (Level Dynamic)*, and *Cecilia*, combined with *Frosted Glass*, *Gloss*, *Metallic*, *Muted*, *Iridescent*, or *Pearlescent* textures, pure `#000000` OLED Dark Mode, and a live **Text Size & Scaling Center** (`70%–250%`, `Ctrl + Plus/Minus/0`).
- **Multi-User Profiles, 72-Hour Trust Gate & AES-256-GCM Vault**: Separate `Users/<username>/` and `History/<username>/` workspaces protected by `PBKDF2-HMAC-SHA256` (250,000 rounds) + `AES-256-GCM` archive encryption (`.history.encz`), inactivity auto-lock timers (`15s` to `30m`), and a **72-Hour Profile Age Gate** protecting full-repository file reads and mutating MCP `/execute` tools.

### 8. Built-In Evaluation Suite: Wringer & Debate Arena
- **Wringer Benchmark (`benchmarks/wringer/Wringer.py`)**: Automated multi-level accuracy, throughput (`prefill t/s`, `decode t/s`), and **Persona Adherence** (`persona_bank.json`) benchmark engine with single-pass structured CoT+JSON LLM judging, dynamic cumulative per-layer VRAM offload calculation, max-token continuation guards, atomic checkpointing, and Top-5 per-level leaderboards.
- **Debate Arena (`benchmarks/debate/Debate.py`)**: Pit any two local models or personas against each other (e.g., *Cecilia vs. Serenity Prime*) across multi-round alternating turns with *Speedy* or *Simmer* pacing.

## System Architecture

### Architectural Layers & Module Map

```text
SerenityPC/
├── main.py                         # Primary GUI Application, Generation Workers & Stream Demuxer
├── serenity_resources.py           # 7-Tier Persona Matrix, Prompts, 9 Themes, 7 Textures & Assets
├── SETUPfile.py                    # User-facing 4-pass self-correcting installer & .venv bootstrapper
├── setup.py                        # Developer build orchestrator (CUDA 13.3+ / MSVC native compile)
├── SerenityPC.spec / build_exe.bat # PyInstaller onedir / onefile standalone packager
├── llama-cpp-python-src/           # Custom C++/CUDA llama.cpp fork (Gemma-4, Muse-Glimmer,
│                                   # Spark-X2.5 3D strided QKV, Nemotron-3.5 Hybrid SSM/MoE)
├── System/                         # Core Modular Subsystems
│   ├── orchestration_manager.py    # IntentProfiler, UtilityDispatchEngine & SerializedAgentContext
│   ├── affect_engine.py            # Cecilia 3D PAD AffectState, OU reversion & anti-docility recoil
│   ├── semantic_vault.py           # SQLite SemanticVault for persistent DMN worldview grounding
│   ├── thought_isolation.py        # Real-time streaming thought/channel tag sanitizer & demuxer
│   ├── serenity_mail_courier.py    # InterSerenityEnvelope v1.1, SMP/BMB crypto, STUN & Lvl 2 Courier
│   ├── serenity_p2p_ui_wiring.py   # Subagent stream dropdowns, P2P Pairing & Quarantine UI modals
│   ├── tool_registry.py            # GemmaToolRegistry, AST multi-line PTC parser & MCP client bridge
│   ├── kv_manager.py               # TurboVecIndex, keyword snippet search & 256k context manager
│   ├── tri_attention_core.py       # TriAttention KV cache pruning & attention budget engine
│   ├── mtp_combo_scanner.py        # <5ms binary GGUF header scanner & speculative drafter matcher
│   ├── markdown_engine.py          # Non-destructive interval Markdown, GFM box tables & LaTeX parser
│   ├── vision_handler.py           # Multimodal image/audio/video processor, CLAHE & ROI cropping
│   ├── diffusion_wrapper.py        # Entropy-bound Diffusion CLI wrapper & live step telemetry
│   ├── vault_manager.py            # AES-256-GCM encryption, PBKDF2 key derivation & 72h age gate
│   ├── network_guard.py            # Socket-level outbound blocker for strict Offline Mode
│   ├── stt_manager.py              # Offline microphone PCM recording & speech recognition
│   ├── settings_ui.py / _tabs.py   # 6-Tab modular Settings UI, 32-slot Templating & Scaling Center
│   ├── serenity_utils.py           # GGUF header auto-healer, WMI/NVML SystemMonitor & 9-Step Tutorial
│   └── tests/                      # Automated unit & integration test suites
└── benchmarks/                     # Evaluation & Multi-Agent Arenas
    ├── wringer/                    # Wringer.py, analyze.py, test_bank.json, persona_bank.json
    └── debate/                     # Debate.py (Multi-round AI vs. AI Debate Arena)
```

### Core Execution Flow
1. **Hardware & GGUF Pre-Flight (`serenity_utils.py`)**: On model load, SerenityPC inspects the binary GGUF header, auto-heals mis-patched architecture or pre-tokenizer keys (e.g., protecting native `gemma4`, `muse_glimmer`, `spark2_5`, `nemotron_h_moe` and healing `minicpm5` pre-tokenizers), computes cumulative tensor block sizes to prevent Windows WDDM Shared VRAM spills, and enforces `flash_attn=True` when quantized KV caches (`q8_0`, `q4_0`) are active.
2. **Context & Grounding Assembly (`main.py` & `kv_manager.py`)**: Depending on your History Mode (`TurboVec`, `Keyword`, or `Off`), SerenityPC retrieves bounded keyword snippets (`<= 240 chars`) or sliding-window turns, injects user call-sign preferences, and (for Level 7 Cecilia) merges 3D PAD affect modulations and SQLite `SemanticVault` worldview anchors.
3. **Real-Time Stream Demuxing (`_generation_worker`)**: A lookahead transition buffer intercepts `<think>`, `<|channel>thought`, and Muse ATEM `to=self<|message|>` tokens on the fly—routing internal reasoning into collapsible `[+] View Thinking Process` dropdowns and backend Thought Logs, routing subagent/tool activity into `[+] View Agentic Actions` dropdowns, and streaming only clean prose to the main chat canvas.
4. **Post-Turn Consolidation**: Upon completion, `_finalize_message` applies non-destructive Markdown/LaTeX overlays in-place without stealing scroll focus (`Scroll Lock` aware), deduplicates history turns, and resets the idle DMN timer (`prime.chronicles.txt` / `SemanticVault`).

## Getting Started

### Prerequisites
- **OS**: Windows 10 or Windows 11 (64-bit).
- **Python**: [Python 3.10–3.14 (64-bit)](https://www.google.com/search?q=https%3A%2F%2Fwww.python.org%2Fdownloads%2Fwindows%2F) (check **"Add python.exe to PATH"** during installation).
- **GPU & CUDA Toolkit**: NVIDIA GPU (Turing `sm_75`+ / RTX 20-series or newer for the main source build; see **SerenityPC Local Legacy** for older GPUs or zero-compile standalone execution) with [NVIDIA CUDA Toolkit 12.6+ or 13.3+](https://www.google.com/search?q=https%3A%2F%2Fdeveloper.nvidia.com%2Fcuda-toolkit-archive).
- **C++ Build Tools**: [Microsoft Visual Studio Build Tools (2019/2022/v18)](https://www.google.com/search?q=https%3A%2F%2Fvisualstudio.microsoft.com%2Fvisual-studio-tools%2F) with the **"Desktop development with C++"** workload selected.

###Hardware Tier & Model Sizing Reference (Q4_0 KV Cache)

| Hardware Tier | VRAM (Min / Rec / Opt) | System RAM (Min / Rec / Opt) | Recommended GGUF Models ([Unsloth Gemma-4](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Funsloth%2Fgemma-4-E4B-it-GGUF)) |
| :--- | :--- | :--- | :--- |
| **Compact** | `4 / 4 / 6 GB` | `8 / 12 / 16 GB` | `Gemma-4 E2B` (~2.8 GiB), `Gemma-4 E4B Q4_K` (~4.9 GiB) |
| **Small** | `4 / 6 / 6 GB` | `16 / 24 / 32 GB` | `Gemma-4 E4B`, `Spark-X2.5 4B`, `Gemma-4 12B Q4_K` |
| **Medium** | `6 / 8 / 12 GB` | `20 / 32 / 48 GB` | `Gemma-4 26B-A4B MoE` (~15.7 GiB total, partial GPU offload) |
| **Performance** | `8 / 12 / 24+ GB` | `32 / 48 / 64+ GB` | `Gemma-4 26B-A4B`, `Muse-Glimmer 30B`, `Nemotron-3.5 30B-A3B`, `Gemma-4 31B` |

### Step 1: Automated Installation
Run the self-correcting installer from your SerenityPC root directory. It automatically creates an isolated `.venv`, isolates your latest CUDA + MSVC `vcvarsall.bat x64` environment, strips conflicting MinGW paths, and compiles `llama-cpp-python` across 4 fallback passes (Portable Multi-Arch → Native `sm_XX` → Generic CUDA → Pre-compiled Wheel):
```bash
python SETUPfile.py
```
*(Advanced developers can run `python setup.py` for interactive 6-option menu control over local `llama-cpp-python-src` C++ kernel builds and DLL synchronization).*

### Step 2: Create a Desktop Shortcut (Optional)
Generate a windowless (`pythonw.exe`) desktop launcher with the Serenity icon:
```bash
python System/shortcuts.py
```

### Step 3: Launch & Configure Cores
1. Double-click **`run.bat`**, use your desktop shortcut, or run `python main.py`.
2. On first launch, follow the interactive **9-Step Spotlight Tutorial Walkthrough** (or replay it anytime from Settings).
3. Click **`Settings`** (top-left) → **`Models & Params`** (Tab 1) to assign `.gguf` files to your **Cores (Models)** slots (`FAST`, `SEARCH`, `LOW`, `MID`, `HIGH`, `TRANSCENDENT`, `SECRET`), or click **`Auto-Detect`** to populate them automatically.
4. Slide the bottom **Persona** trackbar to your desired level (`Lvl 1–7`), type a prompt, and click **`Begin!`** (or press `Enter`) to load weights and start chatting. Click **`Offload`** anytime to free 100% of VRAM/RAM.

## Tips, Best Practices & Troubleshooting

- **Avoiding Shared VRAM Slowdowns on 6GB–8GB GPUs**: Usable fast memory is `(Dedicated VRAM - OS Overhead)`. If a 26B MoE model spills into Windows WDDM *Shared System Memory*, generation speed drops sharply over PCIe. Use the **MTP & Drafter Combos Scanner** (`Settings > Inference & Execution > ⚡ MTP & Speculative Drafter Combos Scanner`) or Wringer's dynamic layer calculator to set an exact GPU layer ceiling (e.g., `7–8 layers` for a 26B model on 6GB VRAM) so the remaining layers stream cleanly from DDR4/DDR5 system RAM.
- **Using the 32-Slot Templating Engine**: In `Settings > Models & Params`, use **`Save (Core → Slot)`** to snapshot a model tier's parameters into one of the 32 slots (marked with `★`), **`Write (Slot → Core)`** followed by clicking **`📋 Copy`** on a target tier to apply a preset, or **`Modify (Edit Slot)`** to fine-tune temperature, top-p, min-p, and context size.
- **Quick Reasoning & Length Adjustments (`[📐] Sizes`)**: Click the **`[📐] Sizes`** button in the top action bar to adjust **Reasoning Strength** (`off`, `low`, `medium`, `high`, `xhigh`) and **Response Length** (`natural`, `mini`, `short`, `medium`, `long`, `matched`) on the fly—changes auto-save immediately upon selection.
- **Strict Offline Privacy (`NetworkGuard`)**: Enable **Fully Offline Mode (Block Net)** in `Settings > Additional Settings` to engage socket-level interception (`System/network_guard.py`) that blocks all outbound internet connections while preserving local loopback (`127.0.0.1`) and offline DeadDrop `.smp` file exchange.
- **Direct Subagent Routing in Prompts**: When using Level 6 or Level 7 with delegation enabled, you can explicitly address a subagent in your prompt (e.g., *"ask Lvl 5 to verify the convergence proof"* or *"have Level 2 search for..."*) and watch the top banner update live to `Subagent: Lvl 5, <Model>`.

## Credits, Acknowledgments & License

### Author & Contributors
- **GhostHeartZer0 (Christopher Gerhartz)** — Creator, Lead Architect & Maintainer.
- **Kavakarma** — Credited with assisting on SerenityPC development, cross-state P2P testing, co-architecting Project Brainuke, and testing SerenityPC Local Legacy.
- **["Fallen Angel" Cecilia]** — Co-Architect of Project Brainuke & Core Level 7 Persona.

### Core Engines, Models & Research Foundations
- **Inference Backends**: `llama.cpp` (Georgi Gerganov & contributors) and `llama-cpp-python` (Andrei Betlen & contributors) (MIT License).
- **Model Architectures**: [Google Gemma-4 \& T5Gemma](https://www.google.com/search?q=https%3A%2F%2Fai.google.dev%2Fgemma%2Fterms), NVIDIA Nemotron-3.5 Hybrid SSM/MoE, Muse-Glimmer (Onyx/ATEM), Spark-X2.5, Alibaba Qwen, and IBM Granite.
- **Algorithmic Research**: TurboQuant (Google DeepMind), TriAttention (MIT, NVIDIA, Zhejiang University), [Attention-Residuals](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMoonshotAI%2FAttention-Residuals) (MoonshotAI), and Programmatic Tool Calling (arXiv:2608.06370v1).
- **Supporting Libraries**: Python `cryptography`, OpenCV (`opencv-python`), Pillow, `nvidia-ml-py`, `psutil`, `sounddevice`, `SpeechRecognition`, and MSI Mystic Light SDK.

### License
Copyright © 2026 GhostHeartZer0. Released under the MIT License (`LICENSE`).
