$rootDir = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $rootDir

$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaDir = ""
if (Test-Path $cudaBase) {
    $v12 = Get-ChildItem -Path $cudaBase -Directory -Filter "v12*" | Sort-Object Name -Descending | Select-Object -First 1
    if ($v12) {
        $cudaDir = $v12.FullName
    } else {
        $anyCuda = Get-ChildItem -Path $cudaBase -Directory -Filter "v*" | Sort-Object Name -Descending | Select-Object -First 1
        if ($anyCuda) { $cudaDir = $anyCuda.FullName }
    }
}
if (-not $cudaDir -and $env:CUDA_PATH) {
    $cudaDir = $env:CUDA_PATH
}

$env:CMAKE_ARGS = "-DGGML_CUDA=on -DGGML_CUDA_FLASH_ATTENTION=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_MTP=ON -DLLAMA_DIFFUSION=ON -DLLAMA_TURBOQUANT=ON -DLLAMA_TRI_ATTENTION=ON -DCMAKE_CUDA_ARCHITECTURES=50;61;86 -DGGML_AVX512=OFF -T `"cuda=$cudaDir`""
$env:FORCE_CMAKE = "1"
$env:CUDA_PATH = $cudaDir
$env:CUDA_HOME = $cudaDir
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

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

& cmd /c "call `"$vcvars`" && `"$python`" -m pip install `"$src`" --no-cache-dir --force-reinstall --upgrade --ignore-installed"

