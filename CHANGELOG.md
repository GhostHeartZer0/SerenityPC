# Changelog

## V2.0 Roadmap (Planned Releases)
### Version 2.0.0
- Stable, polished release.
### Version 1.9.0
- Cleared TODO Verifications (Completed Logs)
- Settings Reorganized
### Version 1.8.0
- Cleared TODO list for v2.0
- Deep Cook Verified
### Version 1.7.0
- Subagents Implemented
## Version 1.6.9
- **UI Dynamic Auto-Scaling, High-DPI Awareness & 1080p Layout Alignment**:
  - Implemented `enable_high_dpi_awareness()` in `System/serenity_utils.py` and invoked on startup before `tk.Tk()` initialization to eliminate blurry scaling on Windows high-DPI displays.
  - Normalized inverted and oversized `self.fonts` hierarchy in `main.py` (`small: 9pt`, `main/italic: 10pt`, `bold: 11pt`, `large: 12pt`, markdown fonts: `9-12pt`).
  - Added dynamic persona slider length auto-scaling in `_on_left_resize` (`70-160px`) and tightened padding across persona control widgets to eliminate horizontal text clipping and keep attachment (`+`) and dictation (`🎙️`) buttons accessible at compact resolutions.
  - Resolved telemetry key collision in `_setup_logs_and_stats()` (`"Total VRAM"` and `"RAM"`), refined column paddings, and switched to compact monospace typography preventing value truncation.
  - Added responsive centered default geometry for 1080p displays with `minsize(960, 600)`.
  - Added automated test suite `System/tests/test_ui_scaling_and_dpi.py`.
- **Image Generation & ReAct / JSON Tool Calling Support**:
  - Implemented balanced-brace JSON action block parsing in `main.py` (`_run_tool_loop`), properly intercepting `{"action": "generate_image", "action_input": ...}` ReAct tool calls without falling through to raw text or meta-commentary dumps.
  - Added robust nested argument decoding (handling stringified JSON, single-quoted python dictionaries, and raw prompt strings) in `handle_generate_image` in `System/tool_registry.py`.
  - Replaced hardcoded `python` subprocess call with `sys.executable` in `spawn_viewer` and restyled the HUD overlay with dark theme and electric blue accents.
  - Enabled `generate_image` and `read_file` in `get_definitions` starting at Level 2+.
- **Telemetry Population (VRAM, Total RAM, CPU Temp, CPU Power) Fix & Verification**:
  - Initialized thread-local COM context (`pythoncom.CoInitialize()` / `pythoncom.CoUninitialize()`) in `SystemMonitor._stats_loop` to support reliable background WMI performance counter querying.
  - Implemented `root\cimv2:Win32_PerfFormattedData_Counters_ThermalZoneInformation` non-admin queries for `CPU Temp` with Kelvin to Celsius conversion (`HighPrecisionTemperature` and `Temperature`), with multi-tier fallbacks to `LibreHardwareMonitor`, `OpenHardwareMonitor`, `root\wmi:MSAcpi_ThermalZoneTemperature`, and `psutil`.
  - Implemented `root\cimv2:Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter` RAPL hardware counter queries (`RAPL_Package0_PKG`, `RAPL_Package0_PP0`) for `CPU Power` in milliwatts, with multi-tier fallbacks to hardware monitors and dynamic CPU load-based TDP estimation curve.
  - Resolved `SharedUsage` WMI query failure in `_get_shared_vram_used_bytes` and added non-NVML / integrated GPU adapter counter fallback (`Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory`) for dedicated and shared VRAM.
  - Updated `_update_stats_display` in `main.py` to format both string and numeric power metrics uniformly (`XX.XW`) in standard and graph modes.
- **Text Size & UI Font Scaling Options**:
  - Implemented dynamic proportional font scaling in `main.py` via `apply_text_scale(scale_pct)` (70% - 250%), updating all `tkFont.Font` bindings across chat, markdown, logs, telemetry, and controls in real time.
  - Re-proportioned base font specifications in `BASE_FONT_SPECS`: boosted main chat / prompt / response readability (`main/bold: 13pt`, `large/headers: 15-17pt`, `ui_button/ui_label: 11pt`) while keeping backend console and telemetry compact (`log: 9pt`, `stats: 8pt`).
  - Switched default font family from missing `Open Sans` to native `Segoe UI` (guaranteed on Windows) to prevent silent fallback font distortion.
  - Added Font Family selection dropdowns ("UI Font" and "Code / Log Font") in `System/settings_ui.py` with 14 popular UI choices (Times New Roman, Comic Sans MS, Arial, Calibri, Verdana, Tahoma, Trebuchet MS, Georgia, Cambria, Palatino Linotype, Franklin Gothic Medium, Impact, Lucida Sans Unicode, Segoe UI) and 5 monospace choices (Consolas, Courier New, Lucida Console, Cascadia Code, Cascadia Mono) with live preview and persistence.
  - Eliminated all hardcoded `font=(...)` tuples across `main.py`, `settings_ui.py`, and `serenity_utils.py`, routing all UI elements through `self.fonts[...]` so scale and family changes propagate globally.
  - Added "Text Size / Scale" preset dropdown (`85% (Compact)` to `200% (Maximum)`) in `System/settings_ui.py`.
  - Added global browser-style zoom keyboard shortcuts: `Ctrl + Plus` / `Ctrl + Equal` (Zoom In +10%), `Ctrl + Minus` (Zoom Out -10%), and `Ctrl + 0` (Reset to 100%).
  - Added automated test suite `System/tests/test_text_scaling.py` covering scale hierarchy, boundaries, and font family switching.

## Version 1.6.8
- **Loading Bar & Status Area Verification**:
  - Verified hybrid status tracking, TTFT calculation, active task gauges, and Serenity Prayer animation transitions.
- **Tutorial & Info Hover Verification**:
  - Verified 9-step tutorial overlay alignment, non-blocking alpha darkening, and 1.5s tooltip timers across controls.

## Version 1.6.7
- **Tutorial & Tooltip Enhancements**:
  - Updated Tutorial Step 2 with focused spotlight bounding box wrapping the Top Action Bar (`top_bar_frame`, `tab_bar_frame`, controls, and telemetry badges).
  - Lightened tutorial overlay background darkening (`-alpha 0.72`) for greater clarity and visibility of the underlying application.
  - Replaced Tutorial Step 8 with a complete, structured Settings walkthrough covering hardware allocation, auto-detect, 32-slot templating, KV cache quants, themes, material textures, and AES-256 Vault encryption.
  - Set default `ToolTip` linger delay to 1.5s (`delay_ms = 1500`) across the application.
  - Expanded hover tooltips across words, titles, labels, entries, and controls throughout `main.py` and `System/settings_ui.py`.
  - Fixed undefined variable `stt_lang_var` in `System/settings_ui.py`.
- **Loading Bar & Status Transition Overhaul**:
  - Fixed premature status bar termination in `main.py` (`_update_ai_message` / `_replace_ai_message`) by updating phase to `generating` instead of calling `think.stop()`, allowing loading progress, TTFT/speed gauges, animations, and Serenity Prayer to remain active and smoothly transition throughout stream generation until final completion.
  - Resolved Serenity Prayer animation freeze and blank screen bug in `System/serenity_utils.py` by repairing the phase cycle state machine and implementing smooth contrast color interpolation.
  - Replaced hardcoded static `"Waking up the experts... (SATA speeds, hang tight)"` string in `_load_model()` with dynamic model loading and prefill phases.
- **Settings Window Enhancements & Dialog Hierarchy**:
  - Added dedicated `[Apply]` button to `System/settings_ui.py` allowing instant runtime parameter application without closing the settings window.
  - Attached `win.transient(app.root)` and `t_win.transient(win)` with `parent=win` modal dialogues, preventing the Settings window from falling behind the main application when modifying or applying templates.
- **User Profile Discovery Filtering**:
  - Updated `list_user_profiles()` in `main.py` to exclude internal utility directories (`backups`, `backups_repair`, and `jsonz to txt`).
- **BF16 KV Cache Removal**:
  - Removed unsupported/crashing `bf16` KV cache type from `System/settings_ui.py`, `main.py`, and `System/tests/benchmarks/debate/Debate.py`.
- **Tutorial Cutoff Prevention & Persistence Fix**:
  - Restructured `TutorialOverlay` widget packing in `System/serenity_utils.py` with bottom-pinned action bar (`< Back`, `Skip`, `Next`) and hint banner to prevent vertical layout truncation.
  - Added live window dimension synchronization, bounds clamping, and `<Configure>` resize listener to keep the tutorial card and spotlight highlight aligned inside the window.
  - Persisted `tutorial_completed` flag in `save_config()` and initialized default in `load_config()` in `main.py`, preventing the walkthrough from re-appearing on subsequent startups.
- **STT Trigger Relocation & Label Cleanup**:
  - Moved STT dictation trigger button (`self.mic_button`) to the persona control bar immediately to the right of the `+` attachment button in `main.py`.
  - Streamlined button label to clean microphone emoji `🎙️` (recording `🔴` / dictating `⏳`) removing redundant text word.
- **Tool Calling Graceful Fallbacks & Mass-Dump Fix**:
  - Implemented resilient fallback observations across `web_search`, `read_file`, `get_system_stats`, `control_rgb`, and `generate_image` in `System/tool_registry.py`.
  - Added `streaming_replace` queue dispatch in `_run_tool_loop` in `main.py`, cleanly scrubbing raw tool execution syntax before live-streaming synthesized answers.
  - Enhanced `_finalize_message` in `main.py` to apply markdown in-place over already-streamed content, eliminating post-tool response mass-dumping and flickering.
- **DMN State Idle Timer Accuracy**:
  - Added state-aware `dmn_active` and `dmn_entry_time` timestamp tracking in `set_avatar_state()` in `main.py`.
  - Updated `DynamicStatusWidget._update_idle_display()` in `System/serenity_utils.py` to display exact elapsed time in DMN state (`[DMN Active] Time in DMN: mm:ss` / `[DMN Simmering]`) when in DMN mode, and countdown (`[Idle] Next DMN in: mm:ss`) during normal idle.
- **Tutorial Interactive Section Spotlighting & Adaptive Layout**:
  - Upgraded `TutorialOverlay` in `System/serenity_utils.py` with dynamic target bounding box resolution across all 9 interface areas (`Avatar & Status Area`, `Persona Matrix`, `Console & Speech`, `Reasoning Logs`, `History & Vault`, `Deep Cook & Debate`, `Settings & Tuning`).
  - Added multi-layered glowing neon spotlight frames (`#00ffcc` / `#007acc`) drawn live around active target UI components with contextual section banners (`📍 SECTION IN FOCUS: ...`).
  - Implemented adaptive smart dialog card positioning (top, bottom, left, or center) so highlighted UI sections remain fully unobstructed and visible during walkthrough.
- **Wringer Benchmark Incremental Checkpointing & Auto-Resumption**:
  - Implemented atomic per-prompt and per-level persistent caching in `System/tests/benchmarks/wringer/Wringer.py` saving directly to `{model_name}_checkpoint.json`.
  - Added incremental `(IN PROGRESS - AUTO-CHECKPOINTED)` markdown report export after each level.
  - Enabled auto-resumption skipping already evaluated levels and reusing cached prompt responses on restart.
- **Bugfixes & Startup Initialization Safety**:
  - Imported `shutil` in `main.py` resolving `NameError: name 'shutil' is not defined` during legacy flat history migration.
  - Reordered `ChatbotApp.__init__` container definitions so `self.config` and dict configs are initialized before invoking `_load_dmn_backbone()`.
  - Added defensive `hasattr` validation to `ChatbotApp.get_active_username()` to prevent premature attribute lookup crashes during early initialization.
  - Added `self._load_dmn_backbone()` refresh inside `load_config()` to sync active user mind state on startup and config reloads.
- **Linger-Hover Help Boxes (ToolTips) & UI Descriptions**:
  - Implemented `ToolTip` class in `System/serenity_utils.py` featuring configurable delay timing (`delay_ms=500`), boundary-safe screen placement, theme-aligned styling (`widget_bg_color`, `electric_blue` border), and automatic text wrapping.
  - Added global hover tooltip toggle checkbox (`Enable Hover Tooltips / Help`) to `System/settings_ui.py` persisting to `System/config.json` (`show_tooltips`).
  - Attached contextual tooltips across key UI components in `main.py` and `System/settings_ui.py` including Model Settings, Begin / Model Swap, Video Multimodal, Pulse Watch, Active Chat / History tabs, User Input, Send, Offline STT Dictation, Deep Cook, Halt, Ghost Mode, History Usage, Persona Slider, Attachment Menu, and hardware options.
- **Translucent Tutorial Overlay & 9-Area Walkthrough**:
  - Created `TutorialOverlay` in `System/serenity_utils.py` with multi-screen glassmorphic interface (`-alpha 0.94`), progress tracking bar, and `Step X of 9` badge indicator.
  - Designed distinct walkthrough modules covering all major subsystems:
    1. *Welcome & System Overview*: Local inference, zero cloud telemetry, and isolated memory.
    2. *Avatar & Status Area*: Emotion reflections, generation phases, and DMN idle timer.
    3. *Persona Hierarchy*: Slider navigation across Levels 1-6 and Cecilia secret unlock.
    4. *Input Console & Dictation*: Text prompt, offline STT mic, media attachments, Ghost Mode, and History toggles.
    5. *Reasoning & Thought Channels*: Real-time stream demuxing, collapsible dropdowns, and LaTeX markdown rendering.
    6. *History & Cryptographic Vault*: Multi-user archives, deep search, and AES-256-GCM encryption.
    7. *Deep Cook & Debate Arena*: Multi-round recursive cycles and agent debate pacing.
    8. *Settings & Hardware Tuning*: Layer allocation, Quantized KV caches, and theme customization.
    9. *Quick Start*: Session completion, tooltips, and ready prompt.
  - Added skip and back/next navigation controls with keyboard bindings (`Left`, `Right`, `Enter`, `Escape`).
  - Implemented first-run auto-launch watchdog (`tutorial_completed: false`) on initial startup and added persistent manual launch button (`[🚀 Tutorial Walkthrough]`) to the top action bar in `System/settings_ui.py`.

## Version 1.6.6
- **Loading Bar & Status Area Overhaul**:
  - Replaced static indeterminate ping-pong progress bar with `DynamicStatusWidget` in `System/serenity_utils.py` and `main.py`.
  - Added multi-phase generation tracking (`[Loading Model]`, `[Prefill]`, `[Reasoning / Thoughts]`, `[Generating]`, `[Complete]`).
  - Added percentage gauge tracking TTFT and estimated duration without visual overlap.
  - Added selectable canvas animations (`spinner`, `pulse`, `orbit`).
  - Added smooth Serenity Prayer text fader with long transition pauses.
  - Added DMN idle timer mode and hybrid smart mode displaying finish speed (`t/s`).
  - Added fallback telemetry displaying active persona level, KV cache format, and context size.
  - Added loading bar configuration panel in `System/settings_ui.py`.
- **Username Profile & Mind State Isolation**:
  - Implemented multi-user profile separation with directory namespaces: `History/<username>/` and `Users/<username>/` under Serenity root.
  - Preserved backward compatibility by auto-migrating legacy root archives to `History/Default/`.
  - Added user switching via `switch_user()` in `main.py`, reloading scoped history archives and isolated user `dmn_backbone.json` reflection state.
  - Updated `VaultManager` in `System/vault_manager.py` to recursively scan, encrypt, decrypt, and migrate user subdirectory archives safely with zero data loss.
  - Added User Profile selection and creation UI controls to `System/settings_ui.py`.

## Version 1.6.5-alpha
- **Wringer Speed Telemetry & Graph Popup Overhaul**:
  - **Streaming Tokens/Sec (`t/s`) Telemetry**: Added per-prompt streaming token measurement in `System/tests/benchmarks/wringer/Wringer.py` calculating precise prefill (`t/s`), decode (`t/s`), and overall generation speed (`t/s`).
  - **Outlier / Anomaly Detection**: Integrated interquartile range (IQR) detection (`detect_anomalies_and_stats`) to track speed spikes/drops and report clean means along with an `anomaly_count`.
  - **Separate High-Scores & Speed Weighting**: Separated quality score and decode speed highscore records in `wringer_highscores.json`. Added user weighting toggle (`n` = strictly separate, `y` = 75% quality + 25% throughput composite score).
  - **Redone Graph Popup Handling**: Redesigned chart layouts with dual-axis quality and speed visualization, disabled disruptive auto-popups by default, and added menu actions for on-demand opening of generated charts and report folders.
- **Edge Cases & Button State Fixes**:
  - **Prompt Visibility Preservation During Model Load**: Updated `clear_chat_ui(preserve_pending=True)` to retain pending user prompt in `chat_history` during dynamic model swaps and asynchronous engine loads.
  - **Button State & Color Synchronization**: Updated `set_ui_state()` and `final_initial_setup()` to dynamically synchronize `ghost_button` and `history_usage_button` labels, colors, and disabled states with `self.config` on launch and during inference.
  - **History Search Bar / Pinned Prompt Collision Fix**: Made `show_active_chat()` check whether `prompt_display` contains text before packing, preventing empty grey prompt bars from appearing as residual search bars after switching tabs.
- **Offline Speech-to-Text (STT) Integration**:
  - Created `STTManager` in `System/stt_manager.py` using `sounddevice` for microphone PCM recording and local `speech_recognition` (offline Sphinx, local Vosk, or multimodal LLM ASR fallback).
  - Added push-to-record `[🎙️] Mic` button to UI footer controls in `main.py` with dynamic recording (`[🔴 Rec...]`) and transcription (`[⏳ Dictating...]`) indicators.
  - Added audio input device selection and language dropdowns in `System/settings_ui.py`.
- **Offline Tool Declaration Enforcement**:
  - Updated `get_definitions()` in `System/tool_registry.py` to strictly omit `web_search` and remote internet services from model system prompts and programmatic stubs when Offline Mode is active.

## Version 1.6.4-alpha
- **nvidia-ml-py Official Migration**:
  - Replaced deprecated `pynvml` package in `requirements.txt` with official `nvidia-ml-py>=13.610.0`.
  - Uninstalled deprecated PyPI redirector package from `.venv`, resolving runtime `FutureWarning` deprecation warnings on NVML initialization and GPU telemetry retrieval.
  - Updated system monitoring import diagnostics and warnings in `main.py`.
- **Repeat Detection & Loop Handling Overhaul**:
  - Re-architected `_detect_repetition` in `main.py` to support three configurable operational modes:
    - **`off`**: Fully disables stream repetition checks for unrestrained code generation, data manipulation, and batch tasks.
    - **`lazy` (Default)**: Sanitizes markdown code fences (` ```...``` `) and tool-call signatures (`web_search(...)`, `action:...`, `<ctrl42>call:...`) before checking repetition; raises repetition thresholds (`min_len=80`, `max_repeats=4` across an 800-character window, or 5+ consecutive identical lines) to prevent false-positive inference abortion during programming and multi-turn tool calling (`search hi`, `search low`, `search windows`).
    - **`hyper`**: Strict loop detector (`min_len=35`, `max_repeats=3` across 400 characters) with stall-phrase loop detection (`re-read`, `reread`, `look again`, etc.).
  - Added "Repeat Loop Detection:" setting radio buttons in `System/settings_ui.py` (Global Engine Overrides) and persisted selection to `System/config.json` (`repeat_detection_mode`).
  - Cleaned up duplicate `_run_tool_loop` method definition in `main.py`.
  - Added unit test suite `System/tests/test_repeat_detection.py` validating mode behavior across tool calls, code blocks, and degenerate loops.

- **Fully Offline Mode Toggle**:
  - Built `NetworkGuard` in `System/network_guard.py` with system-level `socket.socket.connect` and `socket.create_connection` interception, blocking outbound external network traffic while allowing local loopback (`127.0.0.1`, `localhost`).
  - Added offline guard checks to `handle_web_search` in `System/tool_registry.py`, preventing remote HTTP requests and browser launching with clear user-facing refusal telemetry.
  - Added network fetch guards to `_fetch_and_populate_media` and `_spawn_media_popup` in `main.py`.
  - Added "Fully Offline Mode (Block Net)" checkbox in `System/settings_ui.py` and `[OFFLINE]` status badge to hardware/status telemetry bar in `main.py`.
- **History Window Refresh States & Offload Void Fix**:
  - Added explicit tab state tracking (`self.active_tab = "active" | "history"`) in `main.py`.
  - Made `clear_chat_ui()` tab-aware: when on the History Archive tab, it preserves `history_menu_frame` and invokes `_render_history_menu()` to update file sizes, timestamps, and newly archived sessions without leaving the window blank upon model offload.
  - Resolved history view disappearance when swapping or offloading models.
- **Theme & Dark Mode Overhaul**:
  - Added comprehensive theme palettes in `serenity_resources.py`: **Apex Dark (Default)**, **Goth / Obsidian Dark**, **Crystal Cavern**, and **Fractal Logic**.
  - Added 6 modular texture styles: `default` (original flat), `gloss` (polished sheen), `metallic` (brushed gunmetal), `muted` (soft matte), `iridescent` (prismatic shimmer), and `pearlescent` (luminous pearl luster).
  - Added **Frosted Glass Texture** toggle modifier for soft diffusion and glassmorphic card backgrounds.
  - Implemented dynamic runtime theme applier `apply_current_theme()` in `main.py` and `apply_theme_to_global()` in `serenity_resources.py` with live Theme & Texture dropdowns in `System/settings_ui.py`.
- **UI Spacing & Scaling Offsets**:
  - Added `root.minsize(960, 640)` enforcing minimum window geometry to eliminate widget overlap on resize.
  - Integrated `main_window` geometry persistence and sash positioning in `save_config_to_file()` and `final_initial_setup()`.
  - Padded and constrained right canvas log container in `_position_canvas_elements` to maintain centered alignment without clipping adjacent widgets.
- **Muse-Glimmer Onyx/ATEM Architecture Alignment**:
  - Mapped `muse-glimmer` architecture string lookup to `LLM_ARCH_QWEN2` in `llama-arch.cpp` and `serenity_utils.py`.
  - Added per-head `attn_q_norm` and `attn_k_norm` RMS normalization in tensor loader (`llama-model.cpp`) and graph builder (`models/qwen2.cpp`) executed prior to RoPE.
  - Aligned sampling stop tokens (`<|end_of_text|>`, `<|eot|>`) and filtered `<|eom|>` in `main.py` to prevent premature generation cutoffs.

## Version 1.6.3-alpha
- **Thought Channel Isolation & Dropdown Fix**:
  - Added missing `<thought>` / `</thought>`, `</|think|>`, `<|im_start|>thought`, and `<|im_end|>` delimiters to inference scout & split logic (`closers`, opener detection, `tag_clean_pattern`) in `main.py` and `_sanitize_synthesis_output`.
  - Fixed issue where Qwen3.8 native `<thought>` tags were stripped but thoughts failed to demux into the UI dropdown due to missing closer delimiters in scout splitting.
  - Added unit test suite in `System/tests/test_thought_isolation.py` verifying thought isolation across Qwen, DeepSeek, Gemma, and ChatML formats.
- **Wringer Benchmark .venv Self-Bootstrap & Click Execution**:
  - Added `_bootstrap_venv()` to Wringer.py before non-standard imports to auto-detect and re-execute within the workspace `.venv` upon file double-click or global Python invocation.
  - Added crash-protection traceback capture in `__main__` to prevent instant terminal closure on unhandled errors.
  - Installed `matplotlib` into workspace `.venv` and added it to requirements.txt.
  - Fixed chart generation in Wringer.py and analyze.py to sort level axes sequentially (`lvl1`..`lvl7`, `carwash_test`) rather than by score descending.
  - Regenerated all 26 model breakdown chart PNGs and consolidated comparison charts.
  - Integrated `split_thoughts_and_answer` into Wringer.py and formatted internal model reasoning into `<details><summary>Reasoning</summary></details>` collapsible blocks in `.md` reports.
  - Retroactively converted all existing `.md` benchmark reports to format reasoning in dedicated `<details>` dropdowns.
- **Thought Channel Protocol Alignment & Nemotron Meta-Loop Fix**:
  - Replaced contradictory meta-restriction prompt injections with clean architecture-aware reasoning directives (`is_nemotron`, `is_qwen`, `is_deepseek`, `is_gemma`) in main.py and Wringer.py.
  - Built real-time streaming thought demuxer in `_generation_worker`: live thinking tokens route directly to the Thought Log (`tool_log_update`), while chat streaming is held until post-closer to eliminate draft preamble and UI rewrite flicker.
  - Added pre-thought draft rollback protection (`streaming_replace`) if speculative text is emitted prior to late `<think>` opening.
  - Added unit test case in test_thought_isolation.py validating draft preamble isolation before thinking blocks.

## Version 1.6.2

- **History Archive Usability **:
  - Rebuilt the History Archive into a unified search and filter interface in `main.py`.
  - Added Level filter dropdown (`All Levels`, `Level 1` through `Level 7`), Date filter dropdown (`All Dates`, `Today`, `Yesterday`, `Past 7 Days`, `Past 30 Days`, `Older`), and Sort dropdown (`Newest First`, `Oldest First`, `Name A-Z`, `Name Z-A`, `Size (Largest)`).
  - Integrated targeted mousewheel scrolling bound strictly to the history canvas on mouse enter/leave, preventing scroll events from bleeding into outer chat/UI widgets.
  - Implemented real-time archive title search alongside deep full-text background search of compressed `.history.jsonz` message bodies, displaying matched dialogue snippets and highlighting occurrences inside conversation view.
- **Vision & Image Recognition Alignment **:
  - Implemented heuristic contour card ROI detection (`crop_active_playing_area`) in `System/vision_handler.py` to isolate card clusters and auto-crop active playing areas from poker/card tables, removing wasted background felt.
  - Added symbol pixel density enhancement (`enhance_symbol_clarity`) utilizing LAB color space CLAHE and unsharp masking to ensure 6 vs 9 numerals and Heart vs Diamond suit serifs remain crisp.
  - Configured high-fidelity Lanczos-4 scaling and 4:4:4 chroma JPEG encoding (`IMWRITE_JPEG_SAMPLING_FACTOR_444`) to eliminate red color bleed on card suits.
- **App Lock & History Encryption **:
  - Built `VaultManager` in `System/vault_manager.py` with AES-256-GCM authenticated encryption and PBKDF2-HMAC-SHA256 (250,000 rounds) key derivation.
  - Implemented transactional batch archive migration (`.history.jsonz` <-> `.history.encz`) backed by automatic timestamped backups and instant full rollback if any verification error occurs.
  - Added loud ALL-CAPS permanent data loss security disclaimer warnings before master password configuration.
  - Created modal startup lock screen (`show_vault_unlock_modal`) and background inactivity watchdog in `main.py` with dual minutes slider, typeable seconds entry, and quick presets (`Off`, `15s`, `30s`, `45s`, `5m`, `15m`, `30m`).
  - Added full Security & Vault Settings UI panel in `System/settings_ui.py`.
- **Pre-Start Splash Realignment**:
  - Re-aligned `LoadingScreen` geometry in `System/serenity_utils.py` to 360x380, constrained avatar thumbnail scaling bounds to 320x270, and raised splash notification text above canvas items to prevent the "Loading... please wait. This'll only take a minute or two." message from overlapping or being cut off by the avatar image.
- **Dynamic Parameter Auto-Adjustment Engine**:
  - Implemented intelligent, in-memory domain-specific sampling adjustments in System/modular_registry.py (`DynamicParamRegistry`) and main.py.
  - Automatically lowers temperature and increases `min_p` for Coding and Math tasks; adjusts parameters dynamically for Creative writing and Factual extractions.
  - Non-destructive: preserves the user's permanent settings on disk and includes a Settings UI toggle (`dynamic_params_enabled`).
- **Modular Registry Pattern**:
  - Built reusable, extensible `ModularRegistry` in System/modular_registry.py supporting decorator-based registration (`@registry.register(key)`), metadata tagging, introspection, and safe execution dispatch.
  - Refactored System/tool_registry.py (`GemmaToolRegistry`) to eliminate monolithic `if-elif` chains.
- **Full Flash Attention Quantized KV Matrix**:
  - Enabled full KV cache quantization suite across System/settings_ui.py, main.py, and Debate.py: `fp16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`, and `f32`.
- **Pip Environment Hygiene**:
  - Validated strict `.venv` isolation against external global python packages (`ai-edge-litert`, `litert-torch`, `foundry-local-sdk`, `qai-hub`).
- **Real-Time Diffusion Visual Denoising & Time-Grounding**:
  - Updated System/diffusion_wrapper.py with dynamic ANSI clear-screen frame parsing, live step tracking (`Denoising: Step X/Y`), real-time step latency calculation, and ETA telemetry.
- **Debate Mode Enhancements**:
  - Built `LoadingSpinner` Canvas animation widget for model loading and generation states.
  - Added **Speedy** vs **Simmer** pacing level descriptors (`max_tokens`, `temperature`, tailored debate system instructions).
  - Fixed multi-round crash in Debate.py by enforcing strictly alternating `user`/`assistant` Jinja message histories.
  - Resolved regression where persistence code was misplaced within `_run_self_analysis()`, preventing the active window from accumulating chat context and causing models to forget previous turns.
- **Markdown, Math & Table Engine Overhaul**:
  - Rewrote System/markdown_engine.py with a direct non-destructive interval-based parser that eliminates all placeholder strings and null bytes (`\x00`), fixing `CODE.` placeholder rendering and clipboard copy-paste cutoff bugs.
  - Implemented GFM table parsing with aligned Unicode box-drawing grids (`┌─┬─┐`, `│...│`, `├─┼─┤`, `└─┴─┘`) and column alignment handling (`:---`, `:---:`, `---:`).
  - Implemented LaTeX-to-Unicode math converter supporting fractions, roots, summations, integrals, Greek characters, superscripts, and subscripts.
  - Added code block isolation: comments (`#`), multiplication (`*`), and variables (`$`, `_`) inside code blocks are completely protected from inline styling.
  - Disambiguated currency (`$100`) from math equations and `snake_case_variables` from italics.
- **Thought Channel Protocol Alignment & Real-Time Stream Demuxing**:
  - Added lookahead streaming buffer in `_generation_worker` in main.py to prevent partial thought tags (`<|channel>`, `<think>`) from momentarily leaking to the Chat UI on Gemma-4 / Qwen models before demuxing kicks in.
  - Fixed thought/answer separation so direct non-thought responses are never mistakenly classified as thinking logs or re-synthesized.
  - Streamlined real-time demuxing to seamlessly route internal reasoning to background buffers while streaming clean final answers.

## Version 1.6.1

- **Engine Tier & Level Mapping Realignment**:
  - Swapped **Secret** and **Live** engine tier slots:
    - **Engine: Transcendent (Lvl 6)**: Formerly "Live", now assigned to Level 6 (The Transcendent One).
    - **Engine: Secret (Lvl 7)**: Assigned to Level 7 (Cecilia evolved unlock).
  - Renamed all `"Live"` tier identifiers across config.json, settings_ui.py, and main.py to `"transcendent"`.
- **Persona Level Hierarchy Swap**:
  - Promoted **The Transcendent One** to standard visible **Level 6** (`PERSONA_DISPLAY_INFO`, `PERSONA_PROMPTS`, `DEEP_COOK_SYSTEM_PROMPTS`, `CONTEXT_SIZE_MAP`), providing seamless out-of-the-box slider access from Level 1 through 6.
  - Re-anchored **Cecilia** as evolved **Level 7** secret unlock persona, triggered via 6-click persona header event.
  - Added dynamic slider auto-hide behavior: slider auto-collapses to `to=6` when navigating to levels 1–6 or upon model offload, expanding to `to=7` only upon secret unlock.
  - Migrated all Cecilia synthesis pipelines (`_perform_level7_synthesis`), generation channels, lore extraction, and dedicated avatar assets (`Cecilia_01.png`) to Level 7 with backwards-compatible aliases.
  - Migrated and swapped all existing chat history archives in History/ between `_lvl6` and `_lvl7`.
  - Fixed slider auto-clamp bug to preserve Level 7 without falling back to Level 6.

## Model Architecture & KV Cache Safety ##
- **Flash Attention All-Quants Compilation Flag**: Updated setup.py and recompiled local `llama_cpp_python` engine with `-DGGML_CUDA_FA_ALL_QUANTS=ON` to ensure Flash Attention kernel coverage across all quantized KV cache configurations.
- **Quantized KV Cache Flash Attention Auto-Enforcement**: Configured `main.py` to automatically enforce `flash_attn = True` whenever quantized KV caches (`q8_0`, `q4_0`) are selected, preventing context creation failures (`Failed to create llama_context`).
- **Muse Glimmer QK-Norm & Gated Attention Graph**:
  - Implemented QK RMSNorm (`attn_q_norm`, `attn_k_norm`) and post-norm (`post_attention_norm`, `post_ffw_norm`) evaluation in `models/llama.cpp` and `models/llama-iswa.cpp`.
  - Added native gated attention evaluation (`wqkv_gate` with sigmoid activation projection) to LLaMA forward graph.
  - Loaded `f_final_logit_softcapping` and `f_attention_scale` in `llama-model.cpp` and applied tanh softcapping in `models/llama.cpp`, resolving attention saturation loops.

## Version 1.6.0

### Core C++ Engine & Gemma-4 Architecture
- **Gemma-4 C++ Kernel & RoPE Alignment**:
  - Reclassified `LLM_TENSOR_ROPE_FREQS` as `LLM_TENSOR_LAYER_INPUT` in `llama-arch.cpp` and updated `gemma4.cpp` to load global `rope_freqs.weight` across all non-SWA layers, restoring correct positional embeddings.
  - Aligned Gemma-4 attention kernel, value vector RMSNorm (`Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)`), LayerScale (`out_scale`), and attention scaling (`f_attention_scale = 1.0f`) with verified working reference (`gemma4-iswa.cpp`).
  - Corrected `per_layer_proj` tensor dimension ordering (`{n_embd_per_layer, n_embd}`) in `gemma4.cpp`, resolving `check_tensor_dims` crash on Gemma-4 E4B models.
  - Removed layer number suffix from non-repeating input layer tensor definitions (`per_layer_model_proj` and `per_layer_proj_norm`) in `gemma4.cpp`, resolving `input/output layer tensor used with a layer number` crash.
  - Registered `tokenizer_pre == "gemma4"` mapping to `LLAMA_VOCAB_PRE_TYPE_GEMMA4` in `llama-vocab.cpp`.
  - Added `il < swa_layers.size()` bounds check in `llama_hparams::is_swa` (`llama-hparams.cpp`), preventing `invalid vector subscript` assertions when loading models with differing SWA patterns or MTP assist models.
- **Native Gemma-4 MTP Assistant & Speculative Decoding**: Added native architecture support for standalone MTP assistant models. Implemented transparent KV fallback resolution (`find_gguf_key_compat`), dynamic `embedding_length_out` / `n_embd_backbone` projection mapping (`nextn.*` and `mtp.*`), automatic KV cache instantiation for standalone draft models, and contiguous view slicing for shared attention keys.
- **Muse Glimmer Native Architecture Mapping & Auxiliary Tensor Tolerance**: Added `"muse-glimmer"` and `"muse_glimmer"` string lookup mappings directly to `llm_arch_from_string` in `llama-arch.cpp`. Updated `llama_model_loader::done_getting_tensors()` in `llama-model-loader.cpp` to tolerate models containing auxiliary/post-norm tensors (`n_created < n_tensors`) without failing on tensor count assertions.
- **NVIDIA Nemotron 3.5 Lightning Hybrid SSM/MoE & Partial RoPE**: Added hybrid Mamba2 SSM + MoE auto-detection in `llama-model-loader.cpp` when models are exported under architecture label `"llama"`. Added transparent key fallback (`llama.*` -> `nemotron_h_moe.*`) for hyperparameters and array keys. Relaxed strict `n_rot == n_embd_head_k` equality checks in `llama-model.cpp`, `models/llama.cpp`, and `models/llama-iswa.cpp` to support partial rotary dimensions (`n_rot: 84` with head size `128`).
- **UI Drag-and-Drop Hook Guard**: Added null and attribute check (`windnd is not None and hasattr(windnd, "hook_dropfiles")`) in `main.py`, preventing startup exceptions on systems without active drag-and-drop extensions.

## Version 1.5.6

### Python Bindings & Ctypes ABI
- **DRY Sampler Ctypes ABI Fix**: Fixed memory alignment corruption in sampler chain by adding `n_ctx_train` (`ctypes.c_int32`) to `llama_sampler_init_dry` signature in `llama_cpp.py` and passing `llama_model_n_ctx_train(model.model)` in `_internals.py`.
- **Penalties Sampler ABI Fix**: Fixed ABI signature mismatch for `llama_sampler_init_penalties` in `llama_cpp.py` and `_internals.py` by removing obsolete leading `n_vocab` argument, eliminating infinite single-token repeat spam (`1111...`, `IIII...`).
- **Context Params Struct Alignment**: Aligned Python `llama_context_params` `_fields_` with upstream `llama.h`. Removed obsolete `n_outputs_max`/`ctx_other` fields and inserted missing `ctx_type`, resolving memory offset corruption and `"Unsupported ctx type"` failures.
- **Ctypes Dynamic Symbol Resolution & Safe Deallocator**: Wrapped `getattr(lib, name)` in `_ctypes_extensions.py` within `llama_cpp` with exception handling for missing/optional exported symbols, preventing fatal `AttributeError` import crashes on customized `llama.cpp` builds. Added safe attribute guards to `LlamaModel.close()` in `_internals.py` to prevent deallocator crashes during cleanup of partially initialized models.

## Version 1.5.5

### Inference, Reasoning Channels & Samplers
- **Gemma-4 Thought Channel & Clean Answer Delivery**: Verified thought channel extraction and separation (`<|channel>thought...<channel|>`) routing reasoning steps into the UI Thought Log and final answers (`final_answer`) directly to the active Chat tab with zero tag bleed. Added `r'<\|?turn\|?>'` across all structural tag stripping patterns in `main.py`.
- **Native Embedded Jinja Template Integration**: Unified chat generation and Deep Cook pipelines in `main.py` through `self.model.create_chat_completion(...)`. Eliminated manual raw string token concatenation and `<|think|>` system pollution, allowing models to use their native GGUF embedded Jinja template (`tokenizer.chat_template`).
- **Programmatic Tool Calling (PTC) & Clean Stubs (arXiv:2608.06370v1)**: Transitioned tool declarations from verbose Gemma pseudo-JSON tags to typed Python stubs via `get_python_stubs()` in `System/tool_registry.py`. Upgraded `_run_tool_loop` and `_generation_worker` in `main.py` to parse and execute Python function calls with full post-inference tag sanitization.
- **Sampler Parameter Stabilization in `params.json`**: Reset sampler penalties to recommended baseline (`repeat_penalty: 1.0`, `presence_penalty: 0.0`, `frequency_penalty: 0.0`, `temperature: 0.8`, `min_p: 0.05`) in `System/params.json`. Removed harmful `+1.0` logit bias injection from `main.py`. Updated `load_params()` to auto-populate `params.json` on model load.
- **KV Cache Memory Reset & Format Restriction**: Fixed state leakage where previous session tokens remained in VRAM by enforcing clean sequence reset (`seq_id = -1`) in `_internals.py`. Restricted KV cache format options strictly to verified universal formats (`fp16`, `q8_0`, `q4_0`), purging broken/deprecated formats (`q5_1`, `turbo3_tcq`, etc.).
- **Speculative Drafting Safety & Drafter Auto-Discovery**: Set speculative drafting to default off (`speculative_drafting: false`) with live tier reload upon toggle in Settings UI. Removed silent `LlamaPromptLookupDecoding` fallback in `main.py` when no assistant model is loaded. Added automatic detection for `dflash` and `drafter` keyword filenames in `main.py`.
- **Vision Projector Guard**: Prevented `Llava15ChatHandler` from attaching to `self.model.chat_handler` during non-multimodal text inference, preventing text context corruption.

## Version 1.5.4

### Toolchain, Build Orchestration & GPU Acceleration
- **CUDA 13.3+ MSVC Toolchain Auto-Discovery**: Integrated `get_msvc_env()` to auto-discover and load Visual Studio MSVC environment (`vcvarsall.bat x64`) across `setup.py` and `SETUPfile.py`. Sanitized PATH by stripping conflicting MinGW/w64devkit compilers and legacy CUDA versions.
- **Fast Parallel CUDA Compilation & Native GPU Auto-Detection**: Optimized build pipeline by auto-detecting local GPU compute capability (`nvidia-smi`) and setting `CMAKE_BUILD_PARALLEL_LEVEL` to all available CPU threads with live verbose progress.
- **Runtime Library Synchronization**: Updated `setup.py` to automatically deploy all 12 compiled DLLs and libraries directly to `.venv\Lib\site-packages\llama_cpp\lib` post-build.
- **Setup Orchestrator & Deployment Separation**: Separated developer personal setup (`setup.py`, git-ignored) from user-facing deployment (`SETUPfile.py`, git-tracked) with 4-pass self-correcting fallback installation loop.
- **TurboVec 3-Way Mode Toggle**: Added 3-way control (`on`, `fallback`, `off`) for the TurboVec history indexing subsystem in `settings_ui.py` and `kv_manager.py`, allowing bypass of heavy PyTorch/transformers dependencies in fallback mode.
- **Dynamic CPU Thread Allocation**: Added `HardwareProfile.get_optimal_threads()` to calculate thread counts dynamically based on physical/logical core counts, replacing hardcoded thread counts across model loaders.

## Version 1.5.3
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

