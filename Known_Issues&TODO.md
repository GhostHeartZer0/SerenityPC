# Known Issues

1. **Quantized Model Download / Missing Checkpoint Setup**
   - The Lite runtime defaults to `Models/gemma-3-4b-it-qat-UD-Q4_K_XL.gguf`. Users must manually download or place GGUF weights in the `./Models/` directory before initial launch.

2. **Tkinter & Windows Display Constraints**
   - Splash/loading screen and custom window title bar positioning rely on standard Tkinter frame controls which may require DPI scaling adjustments on high-DPI Windows displays.
   - Lack of fallback native notification sounds on non-Windows platforms if ported outside of Windows runtime.

3. **Virtual VRAM / Resource Floor Limits**
   - Offloading full layers (`n_gpu_layers = -1`) requires sufficient physical VRAM (4GB+ recommended). Low VRAM systems without fallback CPU layer tuning may encounter memory allocation errors during initial context creation.

4. **Multi-Platform Runtime Binary Bundling**
   - Pre-bundled CUDA DLLs in `Runtime/` are targeted specifically for Windows x64 environments with NVIDIA GPUs. Linux/macOS users need to supply native `llama.cpp` shared libraries (`libllama.so` / `libllama.dylib`).

---

# TODO

1. **Automated Checkpoint Downloader**
   - Integrate an interactive Hugging Face model downloader directly into `setup.py` / startup flow to automatically pull recommended Gemma-3 GGUF quantized models.

2. **Cross-Platform Library Isolation**
   - Wrap Windows-specific process management and DLL loading routines with dynamic platform checks (`sys.platform`).
   - Add native Metal (macOS) and ROCm/CPU-only (Linux) binary loading fallbacks.

3. **Dynamic Hardware Detection & Layer Auto-Tuning**
   - Auto-calculate optimal `n_threads` based on physical CPU cores (`psutil.cpu_count(logical=False)`).
   - Dynamically calculate safe GPU layer offloading count based on detected VRAM headroom instead of static `-1` offload defaults.

4. **UI Refinements & Theme Customization**
   - Add UI toggle controls for system font scaling, dark/light theme switching, and inline log viewer line wrapping.
   - Implement customizable persona sliders and system prompts directly inside the Lite control panel UI.
