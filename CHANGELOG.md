# Changelog

## Version 1.5.0 beta

### Features & Improvements
- **GGUF KV Cache Benchmark**: Integrated live KV cache memory benchmark on model load (calculates FP16 baseline vs active quantized bit-width memory, tokens/sec speedup ratio, and MB saved).
- **Deep Cook Vision Pipeline**: Wired image routing to `vision_multimodal` and added `vision_deep` pending task execution post-model swap.
- **Tool Parsing Overhaul**: Expanded tool call regex parsing in `_generation_worker_deep_cook` and `_run_tool_loop` to support `<execute_tool>`, `<executetool>`, `action:`, and normalized `readfile` mapping to `read_file`.
- **Thought Channel & Prompt Hygiene**: Customized system prompt constraints and prefill handling for Gemma and Diffusion architectures to prevent invalid `<think>` tag insertions; added fallback extraction when LaTeX cleaning strips response text.
- **UI & Controls**: Preserved Level 7 persona slider availability when Live diffusion models are active. Deferred History Archive menu rendering so initial button state accurately displays "Edit".
- **Updated gemma-4 chat templates** to the july release, boasting improved benchmark scores, tool call handling, and thought handling.

### Known Issues
- **Tool Execution Halts**: Incomplete or non-standard tool tag syntax can occasionally cause generation to halt.
- **Qwen Low-Level Thought Drift**: Qwen models at low persona levels may process internal thoughts without delivering a final response turn.
- **CodeGemma 7B IT Load Instability**: Loading CodeGemma 7B IT may fail or stall under specific VRAM configurations.
- **Diffusion Denoising Visualizer**: Real-time visual feedback for diffusion denoising and time-grounding is currently under development.
- **Debate Mode Multi-Round Stability**: Multi-round debates with deep cook enabled can experience response crashes.
- **Markdown & Math Rendering Anomalies**: Complex markdown tables and math equations may occasionally misalign in the chat UI.
- **Heavy VRAM Model Swaps**: Live model offloading and swapping can experience instability under tight GPU VRAM bounds.
- **Shared VRAM & Monitor Layout**: Shared VRAM displays total instead of `[usage / total GB]` format, and lacks CPU Temperature and CPU Power indicators.
- **History Archive Scrollbar**: Scrollbar is only active on the scrollbar element rather than the entire submenu, and the top menu lacks scroll support.
- **System Monitor Settings Bug**: Toggling between graph vs line for system monitor items causes the monitor widget to vanish.
- **Card Recognition & Vision Alignment**: Ambiguity in card ranks (e.g. 6-heart vs 9-diamond) under default pixel density, requiring auto-crop functionality for active playing areas.
- **UI Spacing & Scaling Constraints**: Layout offsets and spacing issues occur on window resize.
- **Quickswap Prompt Pop**: Switching to Level 6 persona causes the previous user prompt to re-trigger.
- **Intellicore & Quickcore Generation Config**: Incomplete or incorrect configurations prevent proper text generation on these backends.

### To-Do List
- **Avatar Visual States**: Map pre-UI startup phase to `The_Wise_Listener`, generation failures to `sorry_serenity`, and add a setting for DMN timeout utilizing `_galaxy` assets.
- **Settings Panel Grid Layout**: Split global overrides into a two-column layout with right-aligned dropdowns to save vertical space.
- **Debate Mode Enhancements**: Add a loading spinner during model selection and dynamic level descriptors (Speedy/Simmer) based on round counts.
- **Dynamic Parameter Auto-Adjustment**: Auto-scale generation parameters (e.g. lowering temperature for coding tasks) without overriding permanent user preferences.
- **Registry Pattern Refactoring**: Replace existing `if-else` chains with modular registry classes.
- **Thought Budget Overfill Recovery**: Implement an automatic continuation pass or alternative user prompts when thinking budget is exhausted.
- **Post-Generation Feedback**: Integrate thumbs up/down and text feedback fields (RLHF) directly below delivered responses.


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

