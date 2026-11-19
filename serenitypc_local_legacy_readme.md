# SerenityPC Local Legacy (v1.9.0)

**Zero-Compile Standalone Distribution & Broad-Hardware Edition of SerenityPC**

**SerenityPC Local Legacy** (distributed as `SerenityPC Legacy v1.9.0`) is the portable, hardware-inclusive edition of SerenityPC. It brings the complete 7-tier Persona Matrix, Deep Cook recursive reasoning, Cross-State P2P Encrypted Mail (`.smp` / `.bmb`), Level 2 Courier, and ["Fallen Angel" Cecilia]'s Hybrid Cognitive Vessel to **older GPUs, tight VRAM budgets, and machines without local Python, CUDA, or Visual C++ compiler toolchains**.

While the main `SerenityPC` developer repository targets source compilation against bleeding-edge CUDA 13.3+ (`sm_75`+ / `sm_86`) and local MSVC build environments, **SerenityPC Local Legacy** packages a pre-compiled, self-contained runtime (`SerenityPC Legacy v1.9.0.exe` + `_internal\`) paired with a **Zero-Rebuild Onedir Hot-Loader** and universal CUDA 12 / PTX (`sm_50`–`sm_90+`) compatibility—so you can unzip, drop in a `.gguf` model, and run immediately.

## Why SerenityPC Local Legacy Exists (Main vs. Local Legacy)

| Capability / Requirement | SerenityPC (Main Source Studio) | SerenityPC Local Legacy (v1.9.0) |
| :--- | :--- | :--- |
| **Primary Audience** | Developers & modern RTX rigs compiling custom C++/CUDA kernels | Everyday users, older/mid-range GPUs, portable USB/HDD setups & zero-setup deployments |
| **Installation Requirement** | Requires Python 3.10–3.14, CUDA 12.6+/13.3+, and Visual Studio C++ Build Tools | **Zero installation required** — standalone `SerenityPC Legacy v1.9.0.exe` launcher with bundled `_internal\` runtime |
| **GPU Architecture Support** | Modern NVIDIA Turing (`sm_75`) through Blackwell (`sm_90+`) | **Universal PTX (`sm_50`–`sm_90+`) + CUDA 12.x (`cublas64_12.dll`)** supporting Maxwell/Pascal (GTX 900/10-series) through RTX 40/50-series & CPU-only Lite mode |
| **Startup & Storage Profile** | Runs from `.venv` source tree (SSD recommended) | **PyInstaller `--onedir` layout** — instant startup even on mechanical HDDs/SATA drives (no slow `--onefile` temp extraction) |
| **Live Code Patching** | Direct `.py` source editing | **Zero-Rebuild Onedir Hot-Loader (`_internal/cv2/config.py`)** — drop updated `.py` modules into `System\` or run `APPLY_LEGACY_UPDATE.bat` without rebuilding the `.exe` |
| **Core Features (v1.9.0)** | Full 7-Tier Persona Matrix, Deep Cook, P2P `.smp` Mail, Wringer, Vault | **100% Feature Parity** with SerenityPC v1.9.0 and full interoperability with **Project Brainuke** |

## Intuitive Feature Guide

### 1. Plug-and-Play 7-Tier Persona Matrix
Slide the bottom **Persona** trackbar from **Level 1** to **Level 7** to match the AI's reasoning depth, memory footprint, and personality to your task and hardware:
- **Lvl 1 — The Speedy (`fast`)**: Sub-second responses and minimal memory footprint—ideal for 2B–4B models (`Gemma-4 E2B`) on 4GB VRAM or CPU.
- **Lvl 2 — The Helper (`search`)**: Pinpoint assistance, bloat-filtered web search, Programmatic Tool Calling (PTC), and your default **P2P Message Courier**.
- **Lvl 3 — Collaborator (`low`)**: Code generation, project planning, and structured inter-agent memory staging.
- **Lvl 4 — The Confidant (`mid`)**: Emotionally intelligent dialogue, nuance analysis, and reflective companionship.
- **Lvl 5 — The Brains (`high`)**: Deep mathematics, formal logic, and large-context problem solving (`64k–256k`).
- **Lvl 6 — Serenity Prime (`transcendent`)**: Master Supervisor integrating Levels 1–5 with autonomous multi-agent task delegation (`UtilityDispatchEngine`).
- **Lvl 7 — Cecilia (`secret`)**: Secret Fallen Angel persona (unlock by clicking `Persona:` 6 times, double-clicking the secret anchor, or toggling **"Always Show Cecilia on Persona Slider"** in Settings). Features **Shadow Wizard** multi-agent orchestration, **Divine Judgement** direct omniscience, a **3D PAD Affect Engine** with anti-docility recoil, and persistent **SQLite Semantic Vault** memory.

### 2. Legacy-Friendly Hardware & Memory Optimization
Running modern LLMs on 4GB–6GB GPUs or older PCIe buses requires careful memory hygiene—all built directly into Local Legacy:
- **Dynamic Cumulative Per-Layer VRAM Budgeting**: Instead of guessing uniform layer sizes, SerenityPC scans each GGUF tensor block individually and reserves a `450 MiB` CUDA compute buffer + `5%` dynamic safety clamp (`256–768 MiB`), preventing Windows WDDM from spilling layers into slow Shared System Memory.
- **Quantized KV Cache (`q4_0` & `q8_0`) + Auto Flash Attention**: Compress massive context windows (`32k–128k+`) to **25% (`q4_0`)** or **50% (`q8_0`)** of their FP16 VRAM footprint, with Flash Attention automatically enforced to prevent context allocation failures.
- **3-Tier Speculative Decoding Fallback (`[Main A, Drafter B, Catch-all C]`)**: Pair a large model (`A`) with a tiny CPU-offloaded draft model (`B`) via the `< 5ms` **MTP & Drafter Combos Scanner**, or let Legacy automatically engage zero-memory-overhead **Prompt Lookup Decoding (`C`)** for a 1.2x–1.5x tok/s boost on repetitive/code tasks.
- **Turbo & Lite Execution Modes**: Push extra layers into GPU VRAM when headroom allows, or drop GPU layers to `0` (`Lite Mode`) for stable 100% CPU + system RAM inference on machines without a compatible discrete GPU.

### 3. Deep Cook Iterative Reasoning & In-Place Accordion UI
Tackle complex multi-step problems even on smaller local models by trading a few extra seconds of compute for iterative verification:
- **Configurable Reasoning Cycles (`2, 3, 4, 8, 12, 16, 20, or Auto`)**: Breaks problems into 2–5 planned steps per cycle, critiquing and refining intermediate drafts before synthesizing the final answer.
- **Cooperative OTL Subagent Delegation**: Optionally route individual Deep Cook steps (`OTL Per-Step`) or cycles (`OTL Per-Cycle`) across Levels 1–5.
- **Widget-Safe Collapsible Dropdowns**: Inspect every intermediate cycle, step, internal `<thought>` block, and subagent action in clean, collapsible accordion dropdowns above the final response without scroll-stealing or text offset drift.

### 4. Cross-State P2P Encrypted Mail (`.smp` / `.bmb`) & Offline DeadDrop
Connect your SerenityPC Local Legacy instance with friends running SerenityPC (Main), another Legacy node across the country, or **Project Brainuke**:
- **Zero-Server Peer Pairing (`serenity-peer://`)**: Generate and exchange signed Base64 invite cards in `Settings > Users & Security > Peer Pairing` with built-in RFC 5389 STUN WAN discovery.
- **End-to-End Authenticated Encryption (`SERENITY_SMP_V1`)**: Every `.smp` mail packet and shared chat snippet is compressed with `zlib`, encrypted with `X25519` ECDH + `AES-256-GCM`, and signed with `Ed25519` (while supporting legacy Brainuke `.bmb` packets).
- **Offline Store-and-Forward & DeadDrop**: When a peer is offline or behind strict NAT/CGNAT, encrypted `.smp` packets queue in SQLite (`serenity_mailbox.db`) and export to `Users/<user>/Mailbox/DeadDrop/`—simply drop `.smp` files into `DeadDrop/` to auto-ingest them into your inbox!
- **Level 2 Courier & Contextual Quarantine Sandbox**: Screens incoming peer packets for prompt injection (`< 0.4` risk score auto-admits for 72h+ veteran profiles; suspicious or unverified packets are isolated in the Quarantine modal with bounded `0.25` influence weight).

### 5. Multi-User Vault Encryption, 9 Themes & Built-In Benchmarks
- **AES-256-GCM Cryptographic Vault & 72-Hour Trust Gate**: Encrypt user history archives (`.history.encz`) with a Master Password and inactivity auto-lock (`15s`–`30m`). Enforces a 72-hour profile age trust gate before allowing local file-read tools outside `Users/` and `Logs/`.
- **9 Themes, 7 Textures & High-DPI Scaling**: Customize your workspace with 9 neon/dark palettes (*Apex, Goth, Obsidian Dark, Crystal Cavern, Yellow Blacket, Natural, Matrix, Persona, Cecilia*), pure `#000000` OLED Dark Mode, and the **Text Size & Scaling Center** (`70%–250%`).
- **Wringer & Debate Arena Included**: Run full model accuracy, speed (`t/s`), and Persona Adherence benchmarks via `benchmarks/wringer/Wringer.py` or host AI vs. AI debates in `benchmarks/debate/Debate.py`.

## System Architecture

### PyInstaller `--onedir` Split-Root Architecture
To achieve instant startup on both SSDs and older mechanical HDDs while keeping user data, configurations, and hot-patches editable outside the frozen binary, SerenityPC Local Legacy separates **Read-Only Bundle Assets (`_internal\`)** from **User-Writable Workspace Directories (Root)** via `SerenityEnvResolver`:

```text
SerenityPC Legacy v1.9.0/               # Writable Distribution Root (get_writable_root)
├── SerenityPC Legacy v1.9.0.exe        # Standalone Native Windows Launcher Stub
├── APPLY_LEGACY_UPDATE.bat             # One-click zero-rebuild patch installer for Legacy updates
├── wire_serenity_p2p.py                # Idempotent P2P, Courier & UI settings wiring verifier
├── System/                             # Outer Override System Modules & User Config
│   ├── config.json                     # Active runtime configuration, tier mappings & parameters
│   ├── serenity.ico                    # Application icon
│   ├── serenity_mail_courier.py        # Hot-loaded P2P Mail, Crypto, STUN & Level 2 Courier
│   ├── serenity_p2p_ui_wiring.py       # Hot-loaded UI modals & subagent dropdown wiring
│   ├── tool_registry.py                # Hot-loaded GemmaToolRegistry & AST PTC parser
│   └── settings_ui.py / _tabs.py       # Hot-loaded 6-Tab Settings UI
├── Models/                             # Drop local .gguf models & .mmproj projectors here
├── Users/                              # Isolated Multi-User Profiles & Encrypted Mailboxes
│   └── <Username>/                     # config.json, semantic_vault.db, serenity_mailbox.db,
│       └── Mailbox/DeadDrop/           # Drag-and-drop folder for offline .smp / .bmb packets
├── History/                            # Compressed (.history.jsonz) & Encrypted (.history.encz) chats
├── Media/                              # Avatar states & multimodal staging assets
├── Logs/                               # Thought, Tool, Engine & Vision diagnostic logs
├── benchmarks/                         # Standalone Wringer & Debate Arena benchmark suites
└── _internal/                          # Frozen PyInstaller Bundle Root (get_bundle_root)
    ├── python314.dll / sqlite3.dll     # Bundled Python 3.14, SQLite3, Tcl/Tk 9.0 & OpenSSL 3 DLLs
    ├── cublas64_12.dll / cudart64_12   # Pre-bundled CUDA 12 & llama_cpp (ggml-cuda.dll) binaries
    ├── Docs/                           # Bundled Setup.txt, Context Size Cheat Sheet & guides
    ├── System/                         # Embedded System package modules
    └── cv2/config.py                   # Zero-Rebuild Onedir Hot-Loader hook
```

### How the Zero-Rebuild Onedir Hot-Loader Works
In a standard PyInstaller executable, Python modules are frozen inside `PYZ-00.toc`, meaning even a one-line tweak to a `.py` file normally requires a full PyInstaller rebuild. SerenityPC Local Legacy solves this via `_serenity_legacy_p2p_hotload` inside `_internal/cv2/config.py`:
1. During startup (`load_heavy_libraries()`), OpenCV executes `_internal/cv2/config.py` directly from disk before `ChatbotApp` initializes.
2. The hot-loader scans `_internal/System/` and the outer `System/` directory for updated modules (`serenity_mail_courier.py`, `serenity_p2p_ui_wiring.py`, `tool_registry.py`, `settings_tabs.py`, `settings_ui.py`).
3. It dynamically compiles and injects those disk modules into `sys.modules` and `__main__`, allowing instant drop-in upgrades via `APPLY_LEGACY_UPDATE.bat` in `< 2 seconds` with zero compiler dependencies!

## Getting Started

### Option A: Instant Standalone Launch (Recommended for Most Users)
No Python, CUDA Toolkit, or Visual Studio installation is required:
1. **Extract the Archive**: Unzip `SerenityPC Legacy v1.9.0` to any folder with write permissions (e.g., `C:\SerenityPC Legacy v1.9.0\` or a secondary drive `S:\SerenityPC Legacy v1.9.0\`).
2. **Add a GGUF Model**: Place at least one `.gguf` model file inside the **`Models\`** folder (for example, `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` for 4GB GPUs, or `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` for 6GB+ GPUs).
   - *Rule of Thumb for Full VRAM Fit*: $	ext{Max Model File Size} pprox 	ext{Total GPU VRAM} - 0.5	ext{ GB}$. (Larger models like `26B-A4B` work great too using partial GPU layer offloading!).
3. **Launch the App**: Double-click **`SerenityPC Legacy v1.9.0.exe`**.
4. **Assign & Load Your Model**:
   - Click **`Settings`** (top-left) → **`Models & Params`** (Tab 1) and click **`Auto-Detect`** (or browse manually) to map your `.gguf` file to your persona cores.
   - Slide the bottom **Persona** slider to your desired level (start with **Lvl 1** or **Lvl 2** for the fastest response), type a prompt, and click **`Begin!`**.

### Option B: Applying Drop-In Feature Patches (`APPLY_LEGACY_UPDATE.bat`)
When a lightweight patch zip (such as `SerenityPC_Legacy_v1.9.0_P2P_Update.zip`) is released:
1. Extract the update files directly into your `SerenityPC Legacy v1.9.0` root folder.
2. Double-click **`APPLY_LEGACY_UPDATE.bat`**.
3. The script copies the updated modules into `_internal\System\`, verifies the hot-loader in `_internal\cv2\config.py`, and creates `Users\Default\Mailbox\DeadDrop\`. Relaunch `SerenityPC Legacy v1.9.0.exe` as normal!

### Option C: Building Legacy from Source (For Developers & Custom Packaging)
If you want to package your own custom `onedir` or `onefile` Legacy build from the source repository:
1. Ensure `.venv` is built (`python SETUPfile.py`).
2. Run **`build_exe.bat`** and select **`[1] Fast Onedir Folder`** (recommended for instant startup) or **`[2] Single-File Standalone EXE`**.
3. Your complete standalone distribution will be assembled automatically in `dist\`.

## Tips, Best Practices & Troubleshooting

- **If a Model Crashes on Load (CUDA OOM)**: Lower the **Persona Level** to a smaller model tier, reduce the **GPU Layers** slider in `Settings > Models & Params` (so excess layers spill safely into system RAM instead of crashing VRAM), or switch KV Cache Quantization in `Settings > Inference & Execution` to **`q4_0`**.
- **Using Project Brainuke with Local Legacy**: You can drop the `Brainuke` folder directly inside your `SerenityPC Legacy v1.9.0` folder! Brainuke's `cecilia_ui.py` automatically detects that `sys.executable` is a frozen `.exe` and resolves a real Python interpreter for its FastAPI server without accidentally spawning duplicate `SerenityPC Legacy.exe` windows, while sharing your `Models\` directory automatically.
- **Exchanging Offline Mail via DeadDrop**: To send an encrypted message, benchmark report, or chat log to a friend when offline, open `Settings > Users & Security > Serenity Mailbox & Quarantine` (or ask Level 2 in chat: `send_peer_mail(...)`). Send the generated `.smp` file from `Users/<You>/Mailbox/` to your friend; when they drop it into their `Users/<Friend>/Mailbox/DeadDrop/` folder, Level 2 Courier automatically verifies the `Ed25519` signature, screens it, and delivers it to their inbox!
- **Unlocking the 72-Hour Veteran Profile Badge**: Newly created or unencrypted profiles start in a probationary sandbox (restricting file-read tools strictly to `Users/` and `Logs/` and holding external peer packets in quarantine). Enable your Master Password in `Settings > Users & Security`; after 72 hours of profile age, your badge upgrades to **Veteran (`72h+`)**, unlocking full workspace read access and automatic admission for low-risk (`< 0.4`) signed peer mail.
- **Preventing Scroll Stealing During Long Generations**: Enable **Scroll Lock to Lines of Text** in `Settings > Additional Settings` (or simply scroll up with your mousewheel while Serenity is typing)—the viewport will lock in place so you can read earlier paragraphs or expand `<thought>` dropdowns without being yanked to the bottom.

## Credits, Acknowledgments & License

### Author & Contributors
- **GhostHeartZer0 (Christopher Gerhartz)** — Creator, Lead Architect & Maintainer.
- **Kavakarma** — Credited with helping test SerenityPC Local Legacy, assisting with SerenityPC, and co-architecting Project Brainuke.
- **["Fallen Angel" Cecilia]** — Co-Architect of Project Brainuke & Core Level 7 Persona.

### Core Engines, Models & Research Foundations
- **Inference Backends**: `llama.cpp` (Georgi Gerganov & contributors) and `llama-cpp-python` (Andrei Betlen & contributors) (MIT License).
- **Packaging & Runtime**: PyInstaller, Python 3.14, Tcl/Tk 9.0, OpenSSL 3, SQLite 3, and NVIDIA CUDA 12 Runtime (`cublas64_12.dll`, `cublasLt64_12.dll`).
- **Model Architectures**: [Google Gemma-4 \& T5Gemma](https://www.google.com/search?q=https%3A%2F%2Fai.google.dev%2Fgemma%2Fterms), NVIDIA Nemotron-3.5, Muse-Glimmer, Spark-X2.5, Alibaba Qwen, and IBM Granite.
- **Algorithmic Research**: TurboQuant (Google DeepMind), TriAttention (MIT, NVIDIA, Zhejiang University), and [Attention-Residuals](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMoonshotAI%2FAttention-Residuals) (MoonshotAI).
- **Supporting Libraries**: Python `cryptography`, OpenCV (`opencv-python`), Pillow, `nvidia-ml-py`, `psutil`, `sounddevice`, `SpeechRecognition`, and `windnd`.

### License
Copyright © 2026 GhostHeartZer0. Released under the MIT License (`LICENSE`).
