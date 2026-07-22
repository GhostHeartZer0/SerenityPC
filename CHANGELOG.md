# Changelog

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
