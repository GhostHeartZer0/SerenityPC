# SerenityPC Versioning Specs

This document establishes the official versioning specification for SerenityPC across all documentation, releases, CHANGELOG.md, TODO.txt, and git commit histories.

---

## 1. SemVer 2.0 Historical Version Mapping & Evolution:
- **Version 1.0.0**: Initial 5-level prototype release for local AI inference.
- **Version 1.1.0** *(formerly v1.4.0)*: Initial stable public release (System monitoring, grounding, avatar states).
- **Version 1.2.0** *(formerly v1.6.0)*: Nemotron model family support & C++ backend enhancements.
- **Version 1.3.0** *(formerly v1.5.3)*: Markdown engine optimization, RLHF, setup & .venv handling, MTP.
- **Version 1.4.0** *(formerly v1.6.2)*: History Archive usability, Legacy branch creation for older hardware, backend/setup fixes.
- **Version 1.5.0** *(formerly v1.6.9)*: Local CUDA 13.3+ MSVC toolchain optimization, global text scaling, UI controls.
- **Version 1.6.0**: Muse Glimmer architecture fix & official C++ forward graph backend support.

---

## 2. Near-Term Release Roadmap (Leading to v2.0.0)
- **Version 1.7.0**: Multi-Agent Delegation & Subagents Update.
- **Version 1.8.0**: Stability upgrade & Deep Cook vision pipeline validation.
- **Version 1.9.0**: Verifications complete, test suite passing, and settings reorganization.
- **Version 2.0.0**: Major milestone release (Full stability, validated verifications, complete modernized architecture).

---

## 2. Standardized Semantic Versioning Starting v2.0 (`[MAJOR.MINOR.PATCH]`)

Beginning with `Version 2.0.0`, all versions follow a strict 3-tier structure:

$$\mathbf{v[MAJOR].[MINOR].[PATCH][-SUFFIX]}$$

### A. MAJOR (`v1`, `v2`, `v3`, ...)
- **Definition**: Huge releases with full stability, complete architectural maturity, and all planned verifications executed and verified.
- **Verification**: Manual Verification Required.

### B. MINOR (`x.1`, `x.2`, `x.3`, ...)
- **Definition**: Feature releases, major system upgrades, and milestones achieved when a technical hurdle has been cleared or significant new subsystems are added.
- **Verification**: Auto & Manual Verification.

### C. PATCH (`x.x.1`, `x.x.2`, `x.x.3`, ...) — Even vs. Odd Rule
The third number designates the release scope:
1. **Even Numbers (`2`, `4`, `6`, `8`, ...)** = **Minor Features**
   - Non-breaking capability additions, tool integrations, subagent enhancements, new UI settings controls.
   - Eligible for **Auto-Verification**.
   - *Examples*:
     - `v2.0.2`: Overview DMN (Lvls 6 & 7)
     - `v2.0.4`: Gemma-4 Training Station
     - `v2.0.6`: Subagents / MCP Overhaul
     - `v2.0.8`: Turbo TCQ & KV Cache Resolvers
     - `v2.1.2`: Animated / Living Avatars & 3D Procedural Engine
     - `v2.1.4`: TriAttention & Turbovec History Lookup
     - `v2.1.6`: Attention-Residuals Integration
2. **Odd Numbers (`1`, `3`, `5`, `7`, ...)** = **Bug Patches & Hotfixes**
   - Bug fixes, regression fixes, memory leak resolutions, tokenization/prompt fixes, performance tuning, and stability patches.
   - *Examples*:
     - `v2.0.1`: Initial v2.0 hotfixes & patch
     - `v2.0.3`: DMN timing and token sync bug fix
     - `v2.0.5`: Training station file path resolution fix

---

## 3. Release Stage Suffixes
- `-alpha`: Internal mid-releases and actively iterated developer builds.
- `-beta`: builds undergoing / requiring additional testing.
- `-theta`: a combination of the two.
- `-delta`: needs cleanup/fixing/work on/refactoring.
- `-gamma`: builds awaiting major, specific testing.
