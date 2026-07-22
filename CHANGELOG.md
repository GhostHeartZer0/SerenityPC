# Changelog

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

