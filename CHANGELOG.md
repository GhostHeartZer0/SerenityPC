# Changelog

##Version 1.0.1: Security & System Policy Update
- **Smart App Control & Cache Localization**: Localizes `TEMP`, `TMP`, CUDA compiler JIT cache (`CUDA_CACHE_PATH`), PyTorch build/kernel extensions (`TORCH_EXTENSIONS_DIR`, `PYTORCH_KERNEL_CACHE_PATH`, `TORCH_HOME`), Triton (`TRITON_CACHE_DIR`), Pip (`PIP_CACHE_DIR`), and HuggingFace (`HF_HOME`) into the local workspace `.cache/` directory. Bypasses Windows Smart App Control (SAC) blocks on `%TEMP%` sub-process compilations.

## Version 1.0.0-lite (Initial Lite Release)

### Features & Improvements
- **Streamlined Framework**: Lightweight, portable version of the Serenity AI framework.
- **CUDA Runtime Portability**: Pre-bundled with high-performance CUDA DLLs under `Runtime/` for out-of-the-box NVIDIA hardware acceleration without requiring admin rights.
- **Spectrum Slider**: Full integration of Serenity's persona slider levels specifically mapped to the Gemma-3 architecture.
- **Tkinter Interface**: Custom, clean Tkinter desktop control panel tailored specifically for Lite operations.
- **Setup script**: Included hardware diagnostic and setup scripts for easy GPU linking.
