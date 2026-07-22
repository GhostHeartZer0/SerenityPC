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
