$env:CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FLASH_ATTENTION=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_MTP=ON -DLLAMA_DIFFUSION=ON -DLLAMA_TURBOQUANT=ON -DLLAMA_TRI_ATTENTION=ON -DLLAMA_TURBOVEC=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_AVX512=OFF -T `"cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`""
$env:FORCE_CMAKE="1"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
$python = "c:\Users\ccrg6\SerenityPC\.venv\Scripts\python.exe"

& cmd /c "call `"$vcvars`" && `"$python`" -m pip install llama-cpp-python==0.3.14 --no-cache-dir --force-reinstall --upgrade"
