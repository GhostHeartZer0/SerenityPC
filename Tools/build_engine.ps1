$rootDir = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $rootDir

$env:CMAKE_ARGS = "-DGGML_CUDA=on -DGGML_CUDA_FLASH_ATTENTION=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_MTP=ON -DLLAMA_DIFFUSION=ON -DLLAMA_TURBOQUANT=ON -DLLAMA_TRI_ATTENTION=ON -DLLAMA_TURBOVEC=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_AVX512=OFF -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -T `"cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`""
$env:FORCE_CMAKE = "1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:CUDAFLAGS = "-allow-unsupported-compiler"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    $vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
}

$venvPath = Join-Path $rootDir ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in $venvPath..." -ForegroundColor Yellow
    & python -m venv $venvPath
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    . $activateScript
}

$python = Join-Path $venvPath "Scripts\python.exe"
$src = Join-Path $rootDir "llama-cpp-python-src"

$dist = Join-Path $rootDir "dist"
if (-not (Test-Path $dist)) {
    New-Item -ItemType Directory -Path $dist | Out-Null
}

& cmd /c "call `"$vcvars`" && `"$python`" -m pip wheel `"$src`" --no-deps --no-cache-dir -w `"$dist`""
$latestWheel = (Get-ChildItem -Path $dist -Filter "llama_cpp_python*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
if ($latestWheel) {
    Write-Host "Installing $latestWheel..." -ForegroundColor Green
    & "$python" -m pip install "$latestWheel" --no-deps --force-reinstall --no-cache-dir
}

