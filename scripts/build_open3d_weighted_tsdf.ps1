param(
    [string]$CudaRoot = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    [string]$CudaArchitectures = "86",
    [string]$BuildDirectory = "",
    [string]$InstallDirectory = "",
    [string]$CMakeExecutable = "cmake",
    [string]$VisualStudioInstance = "",
    [string]$ThirdPartyDownloadDirectory = "",
    [int]$BuildParallelism = 4,
    [switch]$Rebuild,
    [switch]$Clean
)
$ErrorActionPreference = "Stop"
# Some restricted launchers omit architecture variables required by nested CMake builds.
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = "AMD64" }
if ($Clean -and $Rebuild) { throw "Choose Clean or Rebuild, not both." }
$repoRoot = Split-Path -Parent $PSScriptRoot
$packageRoot = Join-Path $repoRoot "thirdparty\Open3D-weightedTSDF"
$sourceRoot = Join-Path $packageRoot "source"
$buildRoot = if ($BuildDirectory) { [IO.Path]::GetFullPath($BuildDirectory) } else { Join-Path $packageRoot "build_weightedTSDF" }
$installRoot = if ($InstallDirectory) { [IO.Path]::GetFullPath($InstallDirectory) } else { Join-Path $packageRoot "install" }
$buildProperties = @("/p:VcpkgEnabled=false", "/p:VcpkgEnableManifest=false", "/p:ImportDirectoryBuildProps=false", "/p:ImportDirectoryBuildTargets=false")
if ($Clean) {
    if (Test-Path -LiteralPath (Join-Path $buildRoot "CMakeCache.txt")) {
        & $CMakeExecutable --build $buildRoot --config Release --target clean -- @buildProperties
        if ($LASTEXITCODE -ne 0) { throw "Open3D clean failed." }
    }
    # Keep the install tree: it may be used by other configurations or external consumers.
    return
}
$CudaRoot = [IO.Path]::GetFullPath($CudaRoot)
$nvccPath = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvccPath)) { throw "CUDA Toolkit 11.8 nvcc.exe not found: $nvccPath" }
$nvccVersion = & $nvccPath --version
if ($LASTEXITCODE -ne 0 -or ($nvccVersion -join " ") -notmatch 'release 11\.8,') { throw "Only CUDA Toolkit 11.8 is supported." }
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "CMakeLists.txt"))) { throw "Modified Open3D source is missing." }
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
# Generated Open3D projects must not inherit the framework's manifest/SDK settings.
$localProps = Join-Path $buildRoot "Directory.Build.props"
Set-Content -LiteralPath $localProps -Value '<Project><PropertyGroup><VcpkgEnabled>false</VcpkgEnabled><VcpkgEnableManifest>false</VcpkgEnableManifest><ImportDirectoryBuildTargets>false</ImportDirectoryBuildTargets></PropertyGroup></Project>' -Encoding UTF8

$downloadsRoot = if ($ThirdPartyDownloadDirectory) { [IO.Path]::GetFullPath($ThirdPartyDownloadDirectory) } else { Join-Path $buildRoot "3rdparty_downloads" }
$configureArgs = @(
    "-DCMAKE_CONFIGURATION_TYPES=Release", "-DCMAKE_BUILD_TYPE=Release",
    "-S", $sourceRoot, "-B", $buildRoot, "-G", "Visual Studio 17 2022", "-A", "x64",
    "-T", "cuda=$CudaRoot", "-DCUDAToolkit_ROOT=$CudaRoot", "-DCMAKE_CUDA_COMPILER=$nvccPath",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures", "-DCMAKE_INSTALL_PREFIX=$installRoot",
    "-DOPEN3D_THIRD_PARTY_DOWNLOAD_DIR=$downloadsRoot",
    "-DCMAKE_CUDA_FLAGS=-DWIN32 -D_WINDOWS -D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING",
    "-DCMAKE_VS_GLOBALS=VcpkgEnabled=false;VcpkgEnableManifest=false;ImportDirectoryBuildProps=false;ImportDirectoryBuildTargets=false",
    "-DBUILD_SHARED_LIBS=ON", "-DSTATIC_WINDOWS_RUNTIME=OFF", "-DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON",
    "-DBUILD_CUDA_MODULE=ON", "-DBUILD_COMMON_CUDA_ARCHS=OFF", "-DBUILD_GUI=OFF",
    "-DBUILD_PYTHON_MODULE=OFF", "-DBUILD_ISPC_MODULE=OFF", "-DBUILD_WEBRTC=OFF",
    "-DBUILD_EXAMPLES=OFF", "-DBUILD_UNIT_TESTS=OFF", "-DBUILD_BENCHMARKS=OFF",
    "-DBUILD_AZURE_KINECT=OFF", "-DBUILD_LIBREALSENSE=OFF", "-DBUILD_JUPYTER_EXTENSION=OFF",
    "-DBUILD_PYTORCH_OPS=OFF", "-DBUILD_TENSORFLOW_OPS=OFF"
)

if ($VisualStudioInstance) {
    $instance = $VisualStudioInstance
    if ($instance -notmatch ',version=') {
        $instance = [IO.Path]::GetFullPath($instance).TrimEnd([char[]]@('\', '/')).Replace('\', '/')
        $devenv = Join-Path $instance 'Common7\IDE\devenv.exe'
        if (Test-Path -LiteralPath $devenv) {
            $version = (Get-Item -LiteralPath $devenv).VersionInfo.ProductVersion
            if ($version -match '^\d+\.\d+\.\d+\.\d+$') { $instance += ",version=$version" }
        }
    }
    $configureArgs += "-DCMAKE_GENERATOR_INSTANCE=$instance"
}
& $CMakeExecutable @configureArgs
if ($LASTEXITCODE -ne 0) { throw "Open3D configuration failed." }
$buildArgs = @("--build", $buildRoot, "--config", "Release", "--target", "INSTALL", "--parallel", "$BuildParallelism")
if ($Rebuild) { $buildArgs += "--clean-first" }
& $CMakeExecutable @buildArgs -- @buildProperties
if ($LASTEXITCODE -ne 0) { throw "Open3D build/install failed." }
foreach ($requiredFile in @("bin\Open3D.dll", "lib\Open3D.lib", "include\open3d\t\geometry\VoxelBlockGrid.h")) {
    if (-not (Test-Path -LiteralPath (Join-Path $installRoot $requiredFile))) { throw "Open3D installation is incomplete: $requiredFile" }
}
Write-Host "Modified Open3D installed at $installRoot"
