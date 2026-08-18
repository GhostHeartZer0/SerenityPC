# Changelog

## Cleanup & Maintenance (August 2026)
- **Modular Registry Framework & Dynamic Param Auto-Tuning**:
  - Implemented `System/modular_registry.py` (`ModularRegistry` and `DynamicParamRegistry`) providing thread-safe extensible handler registration and in-memory domain-specific sampling adjustments (coding, math/logic, creative/narrative, factual) without modifying user config files on disk.
  - Added Settings UI toggle (`dynamic_params_enabled`) in `System/settings_ui.py`.
- **Interval-Based Markdown, Math & GFM Table Engine**:
  - Implemented `System/markdown_engine.py` (`MarkdownEngine`) featuring non-destructive direct interval parsing, LaTeX-to-Unicode conversion, aligned Unicode box-drawing table formatting, and protected fenced code block handling without placeholders or null bytes.
  - Upgraded `_apply_markdown` in `main.py` to a single-pass atomic render and added full tag support (`md_header_1..3`, `md_quote`, `md_strike`, `md_math_inline`, `md_math_block`, `md_table`, `md_code`).
- **Programmatic Tool Calling & Modular Execution**:
  - Refactored `System/tool_registry.py` to use `@self.registry.register(...)` decorator pattern.
  - Added typed Python function stubs generator (`get_python_stubs()`) for programmatic tool calling.
- **Denoising Latency & Step ETA Telemetry**:
  - Upgraded `System/diffusion_wrapper.py` with real-time step delta timing, moving-average step latency calculation, and ETA telemetry reporting in the UI thinking status.
- **Universal KV Cache Quantization Matrix**:
  - Enabled full KV cache quantization suite across `System/settings_ui.py` and `main.py` (`fp16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`, `f32`).
- **Thought Channel Streaming Lookahead Buffer & Turn Persistence**:
  - Integrated lookahead streaming buffer in `_generation_worker` in `main.py` to prevent thought tokens (`<|channel>`, `<think>`) from leaking into chat history during prefill/streaming before demuxing.
  - Restored turn persistence (`self.messages.append(...)` and `self.save_history()`) in `_finalize_message()`.
- **Debate Arena Enhancements**:
  - Integrated `LoadingSpinner` rotating Canvas animation widget in `System/tests/benchmarks/debate/Debate.py`.
  - Added Pacing selector (`Speedy` vs `Simmer`) with token limits and temperature profiles.
  - Implemented strictly alternating `user`/`assistant` Jinja message formatting to resolve multi-round template crashes.
- **Engine Tier & Level Mapping Realignment**:
  - Swapped **Secret** and **Live** engine tier slots:
    - **Engine: Transcendent (Lvl 6)**: Formerly "Live", now assigned to Level 6 (The Transcendent One).
    - **Engine: Secret (Lvl 7)**: Assigned to Level 7 (Cecilia evolved unlock).
  - Renamed all `"Live"` tier identifiers across `config.json`, `settings_ui.py`, and `main.py` to `"transcendent"`.
- **Persona Level Hierarchy Swap**:
  - Promoted **The Transcendent One** to standard visible **Level 6** (`PERSONA_DISPLAY_INFO`, `PERSONA_PROMPTS`, `DEEP_COOK_SYSTEM_PROMPTS`, `CONTEXT_SIZE_MAP`), providing seamless out-of-the-box slider access from Level 1 through 6.
  - Re-anchored **Cecilia** as evolved **Level 7** secret unlock persona, triggered via 6-click persona header event.
  - Added dynamic slider auto-hide behavior: slider auto-collapses to `to=6` when navigating to levels 1–6 or upon model offload, expanding to `to=7` only upon secret unlock.
  - Migrated all Cecilia synthesis pipelines (`_perform_level7_synthesis`), generation channels, lore extraction, and dedicated avatar assets (`Cecilia_01.png`) to Level 7 with backwards-compatible aliases.
  - Migrated and swapped all existing chat history archives in History/ between `_lvl6` and `_lvl7`.
  - Fixed slider auto-clamp bug to preserve Level 7 without falling back to Level 6.
- **Fast-Path `.venv` Batch Launcher**: Updated `run.bat` to immediately launch `.venv\Scripts\python.exe` without checking global/system `where python` PATH first, eliminating launch failures on systems with missing global PATH entries or Windows Store Python stubs.
- **Tool Synthesis Thought Separation & Stop Token Alignment**: Added `_isolate_thought_and_response()` in `main.py` to extract reasoning channels (`<|channel>thought...<channel|>`) from tool synthesis outputs into `think_log` and strip residual structural tags from `final_answer`. Filtered `<channel|>` from synthesis stop sequences and expanded token headroom to prevent truncated responses.
- **Automatic `.venv` Activation & Runtime Re-execution**: Added `_ensure_venv()` at top of `main.py` to auto-re-execute with `.venv/Scripts/python.exe` if started from global/system Python. Updated `run.bat` to directly execute `.venv/Scripts/python.exe` after activation. Updated `System/tool_registry.py` to use `sys.executable` instead of `"python"`.
- **Resolved GPU Support (CUDA 12.6 Wheel Bundle)**: Compiled `llama-cpp-python` v0.3.26 with CUDA 12.6 targeting legacy GPU architectures (`sm_50`, `sm_61`, `sm_86`) using VS 2022 toolset (`-vcvars_ver=14.4`). Replaced CPU-only wheel in `wheels/` with the CUDA-enabled wheel (`llama_cpp_python-0.3.26-py3-none-win_amd64.whl`, 288MB). Removed `llama-cpp-python` from `requirements.txt` to prevent pip from overwriting the CUDA binary during dependency installs. Updated `setup.py` console stream encoding to UTF-8 with ASCII indicator fallbacks.
- **Purged TurboVec & Sentence-Transformers**: Replaced heavy vector embedding index (`TurboVecIndex` / `SentenceTransformer`) with a zero-dependency lightweight keyword search index (`HistoryKeywordIndex`). Removed `sentence-transformers` from `requirements.txt`, uninstalled package from `.venv`, purged local wheel bundle, stripped `-DLLAMA_TURBOVEC=ON` compile flags from `Tools/build_engine.ps1`, and updated test suites.
- **CUDA 12.x Priority Targeting & PATH Sanitization**: Updated `System/serenity_utils.py`, `Tools/diag_gpu.py`, and `Tools/build_engine.ps1` to detect and prioritize CUDA 12.x toolkits when newer versions (e.g. CUDA 13.x) are co-installed. Automatically strips conflicting higher-version CUDA entries from runtime `PATH` and registers CUDA 12 `bin` and `lib/x64` DLL directories for legacy GPU inference support.
- **Legacy PC Installation Safeguards**: Added Python environment auto-detection & PATH fallback (`py -3` / `python`) in `run.bat` with clear installation guidance. Streamlined `setup.py` fresh installation messaging for first-time automated setup from local `wheels/`.
- **Pre-Compiled Wheels Bundle**: Exported compiled wheels (`llama-cpp-python` with CUDA, PyTorch, Transformers, Playwright, ChromaDB, etc.) into `wheels/` for instant offline setup on target GTX 1050 Ti / Windows 10 systems without C++ compile waits.
- **Setup & Launcher Automation**: Updated `setup.py` and `run.bat` to detect local `wheels/` and auto-bootstrap `.venv` instantly. Added idiot-proof `README.txt`.
- **Purged Bad Absolute References**: Cleaned hardcoded `C:/Users/ccrg6/...` and `S:/...` local drive references from `System/config.json` and benchmark test scripts, replacing with clean relative workspace paths (`Models/...`).
- **Venv Path Sanitization**: Sanitized `.venv/pyvenv.cfg` command path to use relative `.venv` path for distribution portability.
- **Desktop.ini Removal**: Purged 1,600+ Windows-generated hidden `desktop.ini` files across the workspace.
- **Venv Batch Launcher & Shortcut**: Added `run.bat` to activate `.venv` before launching `main.py`. Updated `System/shortcuts.py` to point the desktop shortcut target to `run.bat`.
- **Git Branch Setup**: Renamed `public` remote branch to `Legacy` (`origin/Legacy`) and set up tracking.

## Version 1.5.1: Legacy Compatibility & Setup Automation
- **.venv Auto-Bootstrapping**: Added `ensure_venv()` to `setup.py` to automatically create `.venv` and re-execute within `.venv` if run from system Python. Ensures `.env` exists with legacy configuration. Automatically installs `pip`, `wheel`, `setuptools`, and `ninja` inside `.venv` for reliable C++ builds. Fixed top-level import ordering (`shutil`, `glob`, `re`). Preserves standard `%LOCALAPPDATA%\Temp` during build execution to prevent Windows ACL `Access is denied` errors when `nvcc` invokes MSVC `vcvars64.bat`.
- **Legacy CUDA Architectures & Build Optimization**: Streamlined CUDA build target architectures to `50;61;86` (Maxwell, Pascal, Ampere), trimming unnecessary targets (`70`, `75`, `80`). Set `CMAKE_BUILD_PARALLEL_LEVEL=4` and `MAX_JOBS=4` to throttle parallel Ninja MSVC/nvcc compilation threads, preventing compiler memory exhaustion and PDB lock timeouts. Implemented `get_short_path()` helper for 8.3 path sanitization.
- **MSVC Environment Injection & VS 2022 Toolset Selection**: Enhanced `capture_vcvars_env()` in `setup.py` to invoke `vcvars64.bat` with `-vcvars_ver=14.4` / `-vcvars_ver=14.3` options, targeting the VS 2022 MSVC toolset headers. This prevents CUDA `cudafe++` parser crash (`0xC0000005 ACCESS_VIOLATION`) caused by attempting to parse VS 2026 / MSVC 19.51 C++ STL headers. Automatically exports `CUDA_PATH`, `CUDA_HOME`, `CUDAToolkit_ROOT`, and `CUDACXX` into `os.environ`.

## Version 1.5.0 beta
### Features & Improvements
- **GGUF KV Cache Benchmark**: Integrated live KV cache memory benchmark on model load (calculates FP16 baseline vs active quantized bit-width memory, tokens/sec speedup ratio, and MB saved).
- **Deep Cook Vision Pipeline**: Wired image routing to `vision_multimodal` and added `vision_deep` pending task execution post-model swap.
- **Tool Parsing Overhaul**: Expanded tool call regex parsing in `_generation_worker_deep_cook` and `_run_tool_loop` to support `<execute_tool>`, `<executetool>`, `action:`, and normalized `readfile` mapping to `read_file`.
- **Thought Channel & Prompt Hygiene**: Customized system prompt constraints and prefill handling for Gemma and Diffusion architectures to prevent invalid `<think>` tag insertions; added fallback extraction when LaTeX cleaning strips response text.
- **UI & Controls**: Preserved Level 7 persona slider availability when Live diffusion models are active. Deferred History Archive menu rendering so initial button state accurately displays "Edit".
- **Main Branch Sync**: Full merge of changes from `main` branch (v1.5.0 beta), including GGUF KV cache benchmarking, vision pipeline updates, tool parsing overhaul, thought channel handling, persona level updates in `serenity_resources.py`, and updated chat templates.

## Version 1.4.2: ML-DSA (Dilithium) & Hardware MAC Binding Security
### Features
- **ML-DSA (Dilithium) PQC Middleware**: Integrated `MLDSAMiddleware` in `t5_server.py` and `setup_engine.py` using `System/pqc_signer.py`. Features 30s sliding window replay protection (`X-PQC-Timestamp`, `X-PQC-Signature`, `X-PQC-Public-Key`, `X-PQC-MAC-Binding`). Updated HTTP requests in `serenity_live.py` and `Serenity_Tray.py` to use `pqcrequest`. Registered `pqcipc_request` in `tool_registry.py`.
- **Hardware MAC Address Key Binding**: Bound key derivation and signature verification directly to host hardware MAC address (`uuid.getnode()`). HMAC/Dilithium signature computed over `timestamp:method:path:mac_hash` with `X-PQC-MAC-Binding` validation in `verifyheaders` to block cross-host forgery.

## Version 1.4.1: Security & System Policy Update
### Features
- **ML-DSA (Dilithium) & Hardware MAC Binding**: Implemented ML-DSA-44 header signing (`X-PQC-Timestamp`, `X-PQC-Signature`, `X-PQC-Public-Key`, `X-PQC-MAC-Binding`) with 30s sliding window replay protection in `System/pqc_signer.py`. Bound signature key derivation and verification directly to host MAC address (`uuid.getnode()`). Added `MLDSAMiddleware` to FastAPI apps (`t5_server.py`, `setup_engine.py`) eliminating static secret strings (`serenity-alpha-core-77X`). Updated client HTTP calls in `serenity_live.py` and `Serenity_Tray.py` to `pqcrequest`, and registered `pqcipc_request` tool.
- **Smart App Control & Cache Localization**: Localizes `TEMP`, `TMP`, CUDA compiler JIT cache (`CUDA_CACHE_PATH`), PyTorch build/kernel extensions (`TORCH_EXTENSIONS_DIR`, `PYTORCH_KERNEL_CACHE_PATH`, `TORCH_HOME`), Triton (`TRITON_CACHE_DIR`), Pip (`PIP_CACHE_DIR`), and HuggingFace (`HF_HOME`) into the local workspace `.cache/` directory. Bypasses Windows Smart App Control (SAC) blocks on `%TEMP%` sub-process compilations.

## Version 1.4.0 (Public Release)

### Features & Improvements
- **Pre-Push Git Logging**: Added a git `pre-push` hook to automatically log commit statistics to `Logs/git_push_log.txt` before pushing.
- **System Monitoring**: Fixed Shared VRAM to show `[usage / total GB]`. Added CPU Temp and CPU Power usage. Replaced disk usage with total VRAM use (Dedicated + Shared) and aligned items evenly. Added a setting to toggle between graph vs line for each system monitor item.
- **Grounding & Relevancy**: Added filename imports for image and document grounding. Included time grounding in relevancy.
- **Avatar Updates**: Made the Serenity avatar more intuitive. Added states like `The_Wise_Listener` for startup, `sorry_serenity` for failed generation, and incorporated `explain_direct` and `explain_wise` based on levels. Preserved `subdued_serenity` for model loading.
- **UI & Chat Interface**: Ensured the original prompt is always visible above thoughts and response, including in ghost mode. Fixed an issue where 'Serenity:' appeared instead of 'Cecilia:'. Cleaned up markdown formatting and optimized handling for tables and math. Added an inline markdown setting.
- **Settings Reorganization**: Moved engines to the top. Grouped dynamic, speculative, and ghost checkboxes. Doubled available templating saves and added a thinking checkbox.
- **New Modes**: Added a new Debate Mode where models are pitted against each other (e.g., 'Cecilia vs The Transcendent One'). Added a "Benchmark?" setting to toggle the loading benchmark.
- **Backend & Caching**: Localized TEMP/TMP and CUDA compiler cache paths to the workspace to bypass Windows security policy blocks. Ensured all subprocess backends (MSVC, CMake, Pip, PyTorch, Triton) respect the localized variables.
- **Models**: Mapped MOE router sizes.

### Fixes
- Resolved licensing issues and added credits.

## Legacy Release (v0.0.1 - ye_olde_serenity)
### Features & Improvements
- **Tkinter Control Panel**: Simple desktop GUI for local chatbot operation.
- **Persona Levels**: Introduced five levels of response complexity and formatting.
- **Model Tiers**: Initial support for low, mid, high, and secret tiers using Gemma-3n-E4B-it.
- **GPU Optimization**: Turbo Mode (+3 GPU layers) and Lite Mode (CPU-only execution).
- **History Compression**: Integrated zlib compression for chat history logs.
- **Widget Logging**: Redirected stderr/stdout logs to the UI control panel.
