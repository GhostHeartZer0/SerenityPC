$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_AVX512=OFF"
$env:FORCE_CMAKE="1"
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
$python = "c:\Users\ccrg6\SerenityPC\.venv\Scripts\python.exe"

& cmd /c "call `"$vcvars`" && `"$python`" -m pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade"
