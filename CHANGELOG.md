# Changelog

## Version 1.5.1: Security & System Policy Update
### Features
- **Smart App Control & Cache Localization**: Localizes `TEMP`, `TMP`, CUDA compiler JIT cache (`CUDA_CACHE_PATH`), PyTorch build/kernel extensions (`TORCH_EXTENSIONS_DIR`, `PYTORCH_KERNEL_CACHE_PATH`, `TORCH_HOME`), Triton (`TRITON_CACHE_DIR`), Pip (`PIP_CACHE_DIR`), and HuggingFace (`HF_HOME`) into the local workspace `.cache/` directory. Bypasses Windows Smart App Control (SAC) blocks on `%TEMP%` sub-process compilations.
- **ML-DSA (Dilithium) & Hardware MAC Binding**: Implemented ML-DSA-44 header signing (`X-PQC-Timestamp`, `X-PQC-Signature`, `X-PQC-Public-Key`, `X-PQC-MAC-Binding`) with 30s sliding window replay protection in `System/pqc_signer.py`. Integrated `MLDSAMiddleware` into `t5_server.py` and `setup_engine.py`. Bound signature key derivation and verification directly to host MAC address digest (`uuid.getnode()`). Updated client request tools across `serenity_live.py`, `Serenity_Tray.py`, and `tool_registry.py`.

## Version 1.5.0-compact

### Compact Branch Initial Commit
- **Codebase Pruning**: Stripped down the repository by removing ~12,000 lines of non-essential code, legacy templates, private logs, and testing benchmarks.
- **Documentation Restructure**: Renamed README.md to Welcome!.txt to act as a user-friendly entry point for users, and updated model descriptions.
- **Cross-Platform Checklist**: Created a workspace TODO.txt detailing steps to remove Windows-specific dependencies (such as `windnd`, `winsound`, VBScript shortcuts, and `taskkill` port management) for future cross-platform compatibility.
- **Model Acquisition Strategy**: Documented and mapped GGUF/projector model equivalents from Hugging Face for local execution.

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
