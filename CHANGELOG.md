# Changelog

## Version 1.5.3

- **Muse Glimmer Native Loader & Custom Build Flags Update**: Updated `setup.py` and `Tools/build_engine.ps1` to inject custom CMake flags (`-DLLAMA_MTP=ON`, `-DLLAMA_DIFFUSION=ON`, `-DLLAMA_TURBOQUANT=ON`, `-DLLAMA_TRI_ATTENTION=ON`, `-DLLAMA_TURBOVEC=ON`). Removed runtime header patching for `muse-glimmer` in `System/serenity_utils.py` to allow native execution via `llama.cpp` build b10353+.

- **Robust Deferred Library Imports**: Guarded `cv2` and `windnd` imports in `load_heavy_libraries()` with fallback exception handling to prevent missing optional libraries from failing core application startup. Added global `numpy as np` initialization in `main.py`, `System/vision_handler.py`, `Wringer.py`, and `Debate.py`. Updated `requirements.txt` with `opencv-python` and `pynvml`.
- **Dynamic CPU Thread Allocation & Safe Core Pinning**: Added `HardwareProfile.get_optimal_threads()` to calculate thread counts dynamically based on physical/logical core counts. Replaced hardcoded `n_threads=8` across standard chat, vision, draft, and auto-recovery model loaders to prevent thread thrashing/OOM on 4-thread and 8-thread CPUs.
- **Muse Glimmer Native Architecture Mapping**: Added `"muse-glimmer"` and `"muse_glimmer"` string lookup mappings directly to `llm_arch_from_string` in `llama-cpp-python-src/vendor/llama.cpp/src/llama-arch.cpp` (mapping to Meta's base `LLM_ARCH_LLAMA`), allowing out-of-the-box GGUF inference without requiring runtime binary header patching.
- **Muse Glimmer & Gemma-4 Architecture Support & Auto-Patching**: Added `patch_gguf_architecture` GGUF binary metadata patcher as a secondary fallback mechanism. Intercepts unknown GGUF model architecture and tokenizer exceptions (such as `muse-glimmer` or `gemma4`) during model initialization and automatically maps Gemma variants (`gemma4`, `gemma3`) to `gemma2`, patching `general.architecture`, `tokenizer.ggml.model` string values, and KV key prefixes (e.g. `gemma4.` -> `gemma2.`) in-place before retrying model load. Added `patch_llama_deallocator` to safely handle partially initialized `LlamaModel` cleanup during failed model loads without throwing `AttributeError`.
- **Muse Reasoning Strength Setting**: Added configurable `muse_reasoning_strength` setting (`off`, `low`, `medium`, `high`, `xhigh`) in the Settings UI with dynamic system prompt injection across standard chat, deep cook, and vision inference loops.
- **Markdown Engine Optimization (TODO #1)**: Refactored `_apply_markdown` to run formatting steps iteratively instead of recursively, optimizing table parsing and math tag rendering to eliminate UI thread pauses.
- **Thought Budget Recovery (TODO #8)**: Added `budget_recovery_mode` setting (`off`, `respond`, `wrapup`, `autocont`) with automated synthesis pass when generation hits token budget within reasoning blocks.
- **Post-Generation RLHF Feedback (TODO #10)**: Embedded 👍/👎 feedback buttons into chat message outputs, saving user ratings to `System/rlhf_logs.json` and integrating stats into DMN backbone memory.
- **Self-Analysis Feature (TODO #12)**: Added "Self-Analysis" status button in Backend Logs header that generates a live configuration status report in the chat window.
- **Legacy PC Hardware & CUDA Setup Auto-Gathering & Upgrades**: Added pre-flight Python bitness/version/RAM checks, CPU AVX/AVX2 capability scans, and CUDA Toolkit / Compute Capability checks to `setup.py`. Implemented `activate_local_venv()` for automatic workspace `.venv` detection with `--global` support for targeting system Python environments, and `gather_missing_tools()` to automatically download, locate, and PATH-inject missing build tools (`cmake`, `ninja`, `git`, `nvcc`, `MSVC`). Configured universal PTX compilation range (`sm_50` through `sm_90+`) to ensure zero-recompile runtime portability on older/other GPUs. Resolved pip file-locking (`WinError 5`) on satisfied requirements and added non-interactive `--rebuild` CLI automation.


## Version 1.5.2

- **MTP Assistant Search**: Added `MTP` keyword matching to MTP assistant model auto-detection.
- **Avatar Aspect Ratio Scaling**: Preserved image aspect ratios when resizing avatar icons (`_fit_image_aspect`).
- **Settings UI Headers**: Titled Engines section (`Text/Inline Engines`).
- **System Monitor Graph Mode**: Fixed system monitor graph vs line mode formatting and display updates.
- **UI Persona Controls**: Set Level 7 to fill slider position 6 directly when Level 6 is hidden.
- **Loading Splash Notification**: Added pre-start message (`Loading... please wait. This'll only take a minute or two.`).
- **History & Chat Cleanup**: Prevented old history messages from popping up in active chat during model loads by keeping active chat fresh on model swap and preserving history archive in Archive tab. Fixed prompt positioning bugs on pending prompt submission.
- **Bidirectional Slider Trap**: Fixed persona slider jumping logic so sliding downwards from level 7 past hidden level 6 correctly lands on level 5.
- **Telemetry Enhancements**: Added `root\LibreHardwareMonitor` namespace queries for CPU Temperature and CPU Power telemetries in addition to `OpenHardwareMonitor` and WMI fallbacks.
- **Git Ignore Hardening**: Added recursive `**/[dD]esktop.ini` and `**/Thumbs.db` pattern rules to `.gitignore`.

## Version 1.5.1

- **Avatar Visual States & Transitions**: Mapped pre-UI startup splash to `The_Wise_Listener`, generation error states to `sorry_serenity` (`apologetic`), prefill phase to `Meditating_Serenity`, response generation to `explain_wise`, and Level 7 persona to `transcendent_serenity`. Configured 3-second transition from `serenity_greeting` to persona idle images, added DMN Timeout setting (`min:sec` format in UI), and fixed model load `pending_task` timer override edge cases.
- **Verified MTP works**: Loaded lvl 7 with gemma-4-26b-a4b and saw a response roughly 5t/s faster.


## Version 1.5.0

### Features & Improvements
- **GGUF KV Cache Benchmark**: Integrated live KV cache memory benchmark on model load (calculates FP16 baseline vs active quantized bit-width memory, tokens/sec speedup ratio, and MB saved).
- **Deep Cook Vision Pipeline**: Wired image routing to `vision_multimodal` and added `vision_deep` pending task execution post-model swap.
- **Tool Parsing Overhaul**: Expanded tool call regex parsing in `_generation_worker_deep_cook` and `_run_tool_loop` to support `<execute_tool>`, `<executetool>`, `action:`, and normalized `readfile` mapping to `read_file`.
- **Thought Channel & Prompt Hygiene**: Customized system prompt constraints and prefill handling for Gemma and Diffusion architectures to prevent invalid `<think>` tag insertions; added fallback extraction when LaTeX cleaning strips response text.
- **UI & Controls**: Preserved Level 7 persona slider availability when Live diffusion models are active. Deferred History Archive menu rendering so initial button state accurately displays "Edit". Added dump icon next to backend log slider to clear active log view.
- **Updated gemma-4 chat templates** to the july release, boasting improved benchmark scores, tool call handling, and thought handling.

## Version 1.4.0

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

