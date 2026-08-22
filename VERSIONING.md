# SerenityPC Local Legacy Versioning Specs

This document establishes the official versioning specification for the **SerenityPC Local Legacy** fork across all documentation, releases, `CHANGELOG.md`, `TODO.txt`, and git commit histories.

---

## 1. Fork Architecture & Branch Point

- **Upstream Origin**: [GhostHeartZer0/SerenityPC](https://github.com/GhostHeartZer0/SerenityPC)
- **Branch Point**: **Version 1.4.0** *(historically tagged as v1.6.2 / v1.4.0 [Stable])*
- **Fork Purpose**: Porting and optimizing SerenityPC for broader consumer hardware, legacy systems, cross-platform compatibility (macOS/Linux/Windows), dynamic hardware allocation, and self-contained model downloads without demanding bleeding-edge C++ engine forks (e.g. CUDA 13.3+ MSVC builds, custom Muse-Glimmer/Nemotron kernels).

---

## 2. SemVer 2.0 Historical Version Mapping (Pre-Branch to Present):

- **Version 1.0.0**: Initial 5-level prototype release for local AI inference.
- **Version 1.1.0** *(formerly v1.4.0)*: Initial stable public release (System monitoring, grounding, avatar states).
- **Version 1.2.0** *(formerly v1.5.0–v1.5.2)*: GGUF KV cache live benchmark, Deep Cook vision pipeline, avatar visual states, MTP assist models.
- **Version 1.3.0** *(formerly v1.5.3–v1.5.6)*: Markdown engine optimization, Thought budget recovery, RLHF feedback logging, self-analysis report, setup toolchain gatherer, PTC typed Python stubs, sampler parameter stabilization.
- **Version 1.4.0** *(formerly v1.6.2 - **LEGACY FORK BRANCH POINT**)*: Unified History Archive search/filter, AES-256-GCM Vault encryption, dynamic parameter auto-tuning, modular registry pattern, quantized KV matrix, Debate mode, LaTeX math / GFM table parser, thought channel demuxing.

---

## 3. Legacy Release Roadmap (Leading to v2.0.0)

- **Version 1.5.0** *(Legacy Core Update)*:
  - Cross-platform portability (guard Windows-only `windnd`, `winsound`, `win32api`, `taskkill`).
  - Removal of hardcoded system paths (relative `./Models/` and user home config).
  - High-DPI awareness, text scaling center, and responsive 1080p UI layout.
  - Upstream UI theme engine (Apex, Obsidian Blackout, Crystal Cavern, Matrix, Persona-Dynamic).
  - Repeat loop detection (`off`, `lazy`, `hyper`) and offline STT dictation integration.
- **Version 1.6.0** *(Hardware Adaptation & Model Hub)*:
  - Dynamic hardware inference targeting (adaptive CPU threads, VRAM floor auto-scaling).
  - Integrated Hugging Face model downloader & tier bucketing.
  - Dynamic Status & 3-line loading bar area with collapsible thought process dropdowns.
- **Version 1.7.0**: Multi-user profile separation (`Users/<username>/`, `Public`, `Default`) and offline guard mode.
- **Version 1.8.0**: Multi-agent delegation & legacy-compatible subagent orchestration.
- **Version 1.9.0**: Settings UI reorganization into dedicated tabs, full test suite pass.
- **Version 2.0.0**: Milestone stable release for legacy & cross-platform deployments.

---

## 4. Standardized Semantic Versioning Starting v2.0 (`[MAJOR.MINOR.PATCH]`)

Beginning with `Version 2.0.0`, all releases follow standard SemVer 2.0 with the Even/Odd patch rule:

$$\mathbf{v[MAJOR].[MINOR].[PATCH][-SUFFIX]}$$

### A. MAJOR (`v1`, `v2`, `v3`, ...)
- Major architectural milestones, complete platform stability, verified cross-platform release.
- **Verification**: Manual Verification.

### B. MINOR (`x.1`, `x.2`, `x.3`, ...)
- Significant feature additions, new subsystem implementations (model downloaders, subagents, audio/vision pipelines).
- **Verification**: Auto & Manual Verification.

### C. PATCH (`x.x.1`, `x.x.2`, `x.x.3`, ...) — Even vs. Odd Rule
1. **Even Numbers (`2`, `4`, `6`, `8`, ...)** = **Minor Features**
   - Non-breaking capability additions, UI widgets, downloader presets, settings controls.
   - Eligible for **Auto-Verification**.
2. **Odd Numbers (`1`, `3`, `5`, `7`, ...)** = **Bug Patches & Hotfixes**
   - Bug fixes, path resolution repairs, cross-platform fallbacks, memory leak patches.

---

## 5. Release Stage Suffixes
- `-alpha`: Internal mid-releases and actively iterated developer builds.
- `-beta`: Builds undergoing broader platform testing.
- `-theta`: Hybrid feature & stabilization builds.
- `-delta`: Code under active refactoring or cleanup.
- `-gamma`: Builds awaiting final platform-specific validation.
