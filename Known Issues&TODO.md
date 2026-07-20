# Known Issues & Workspace TODO

---

## ⚠️ Known Issues & Platform Limitations

### 1. Windows-Specific Hardcoded Logic
- **`windnd` Drag-and-Drop Dependency**: `main.py` directly imports `windnd` without fallback, failing on non-Windows platforms.
- **`winsound` Audio Beeps**: `t5_server.py` and `setup_engine.py` rely on `winsound.Beep`, causing `ImportError` on Linux/macOS.
- **`taskkill` Shutdown Logic**: `kill_engine_on_shutdown()` in `main.py` hardcodes Windows `taskkill /F /PID` for port management instead of cross-platform socket/process checks.
- **VBScript Desktop Shortcut Creation**: `shortcuts.py` and `setup_engine.py` invoke `cscript.exe` to generate `.lnk` files.
- **Windows Subprocess Creation Flags**: `0x08000000` (`CREATE_NO_WINDOW`) is passed to `subprocess` without OS checks (`sys.platform == 'win32'`) in `System/serenity_utils.py` and `Live/Serenity_Tray.py`.
- **Batch Script Launcher**: `run.bat` relies exclusively on Windows shell syntax.

### 2. Path & Environment Hardcoding
- **CUDA/GPU Detection**: `initialize_gpu_acceleration()` in `System/serenity_utils.py` looks primarily for Windows NVCC/CUDA installation paths.

---

## 📋 To-Do List

### 1. Upstream Sync & Main Updates
- [ ] Establish periodic merge strategy from main branch without re-introducing pruned files (~12k lines stripped).
- [ ] Resolve merge conflicts while maintaining compact branch optimizations.

### 2. Cross-Platform Refactoring
- [ ] **`windnd`**: Wrap `import windnd` in `try-except`; gracefully disable drag-and-drop on Linux/macOS.
- [ ] **`winsound`**: Replace `winsound.Beep` with platform-agnostic audio or conditional `try-except` handling.
- [ ] **GPU Detection**: Add `/usr/local/cuda/bin` and macOS Metal/Library paths to `HardwareProfile`.
- [ ] **Subprocess Flags**: Guard `CREATE_NO_WINDOW` (`0x08000000`) with `sys.platform == 'win32'`.
- [ ] **Shortcuts**: Add `.desktop` file creation for Linux and `.app` launcher creation for macOS.
- [ ] **Port Management**: Replace `taskkill` port 8001 termination with `psutil` or socket-level process cleanup.
- [ ] **Shell Launchers**: Add cross-platform `run.sh` bash script parallel to `run.bat`.

### 3. Hugging Face Model Acquisition
- [ ] Document / script model auto-downloads via `huggingface-cli` or `huggingface_hub`.
- [ ] Maintain model mappings for:
  - `gemma-4-26B-A4B-it` (Q4_0, Q8_K_XL, uncensored)
  - `gemma-4-26B-it-mmproj`
  - `diffusiongemma-26B-A4B-it`
  - `gemma-4-E2B-it` / `gemma-4-E4B-it`
  - `Qwen3.6-27B-Instruct` / `Qwen3.6-35B-A3B-Instruct`
  - `codegemma-7b-it`
  - `t5gemma-2` (270m, 1b)

### 4. Codebase Pruning & Optimization
- [ ] Identify and eliminate orphan media assets, obsolete icon sets, and scrap scripts.
- [ ] Audit all path concatenations to mandate `os.path.join` / `pathlib.Path`.

---

