# Known Issues

1. **Windows Platform Coupling**
   - Direct dependencies on Windows-only libraries (`windnd`, `winsound`, `win32api`/`win32con`) prevent execution on macOS and Linux.
   - Process termination relies on Windows-specific `taskkill /F /PID` shell commands in `main.py`.

2. **Hardcoded User & Environment Paths**
   - Absolute local paths (`C:/Users/ccrg6/...`, `S:/...`) exist in `System/config.json` and system path resolution scripts.
   - `setup.py` and resource loaders assume specific local drive letter structures rather than relative workspace or user home directories (`~/.serenity/`).

3. **Static Hardware Tuning**
   - Inference parameters are hardcoded for a specific hardware target (i7-12700KF / RTX 3050 LP): fixed thread count (`n_threads=8`), static batch sizes, and a rigid 500MB VRAM margin check (`nvidia_ml`).

4. **Manual Model Setup Required**
   - Absence of an automated GGUF downloader requires users to manually source and place model checkpoints in `./Models/`.

---

# TODO

1. **Merge & Update from Main**
   - Track upstream commits from main branch ([GhostHeartZer0/SerenityPC](https://github.com/GhostHeartZer0/SerenityPC)).
   - Port bug fixes, UI improvements, and core logic updates while excluding local dev credentials and test scripts.
   - Ensure public release documentation (`README.txt`, `Setup.txt`) remains up to date.

2. **Refactor for Cross-Platform Compatibility**
   - **Tkinter Drag & Drop**: Wrap `windnd` import with `sys.platform` checks or integrate `TkinterDnD2` as a cross-platform fallback.
   - **Audio Notifications**: Guard `winsound` calls in `serenity_live.py`, `t5_server.py`, and `setup_engine.py` with fallbacks (terminal bell, `sounddevice`, or silent fallback).
   - **Win32 API**: Guard `win32api`/`win32con` calls in `serenity_utils.py` with cross-platform alternatives.
   - **Process Termination**: Replace `taskkill` subprocess commands with `psutil` process management.
   - **Build System**: Update `setup.py` to support `gcc`/`clang`, make `vcvars64.bat` optional on Windows, and support Apple Silicon (Metal) and Linux (ROCm/CUDA/CPU) builds.

3. **Remove Hardcoded System Paths & Localization**
   - Convert `System/config.json` absolute paths to relative paths (`./Models/...`).
   - Standardize path resolution across `setup.py`, `main.py`, and `serenity_resources.py` to use relative workspace roots or user home directory (`~/.serenity/`).

4. **Tailor Hardware Inference to Dynamic Targets**
   - Dynamically compute `n_threads` using physical core detection (`psutil.cpu_count(logical=False)`).
   - Dynamically scale `n_batch` and RAM/VRAM allocations according to target system specs.
   - Update `nvidia_ml` VRAM floor checks to adapt dynamically to card sizes (4GB to 24GB+).

5. **Hugging Face Model Downloader Integration**
   - Build an interactive downloader script (`System/model_downloader.py` or integrated into `setup.py`) using `huggingface_hub`.
   - Map levels/personas to GGUF checkpoints (e.g. Gemma 4 GGUF checkpoints).
   - Add hardware assessment to guide users toward the optimal quantization level for their VRAM/RAM capacity.
