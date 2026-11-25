# Modified Open3D 0.18.0 for weighted TSDF

This directory contains a source snapshot of the locally modified Open3D 0.18.0 used by On-the-FlyDSR. The upstream project is https://github.com/isl-org/Open3D/tree/v0.18.0 . Open3D retains its MIT license; this project's root GPL license does not replace it.

- source/: complete local source snapshot, excluding build directories, downloaded dependency caches, bytecode caches and logs.
- source-manifest.json: SHA-256 hashes and sizes of the copied files. No upstream Git commit metadata was present in the local source directory. This is a snapshot, not a verified diff against the upstream tag.
- install/: expected C++ installation prefix, generated locally or extracted from the separately distributed binary archive. Do not commit it.

The weighted TSDF API is in cpp/open3d/t/geometry/VoxelBlockGrid.{h,cpp} and its kernel files. Keep the complete source snapshot until an upstream comparison has established a complete patch set; do not publish only DLL/LIB files as the modification's source.

## Build as part of the solution

Open OntheFlySfM.sln in Visual Studio 2022, select **Release / x64**, and build the solution. The thirdparty solution folder contains Open3DWeightedTSDF. Workflow and UI have explicit project references to it, so Open3D is built and installed before those consumers. No manual PowerShell invocation is needed.

The project invokes scripts/build_open3d_weighted_tsdf.ps1 using the original Open3D CMake project and CUDA Toolkit 11.8. Every build lets CMake check for changed sources; existing objects and downloaded dependencies are reused. Rebuild passes --clean-first. Clean invokes CMake's clean target and retains the install directory. Build directories and downloads are under build_weightedTSDF; installed artifacts are under install. UI copies the installed bin/*.dll files to its output directory, skipping unchanged files.

Only Release/x64 is supported by this integration. CMake, VS 2022 C++ tools and CUDA 11.8 must be installed. The first build downloads Open3D dependencies and requires network access. Default CUDA architecture 86 matches the original build; override Open3DCudaArchitectures for your GPU. Open3DWeightedRoot controls the installation directory. Open3DBuildDirectory and Open3DCMakeExecutable can also be set as MSBuild properties.

For example, from a VS developer shell:

~~~powershell
msbuild OntheFlySfM.sln /m /p:Configuration=Release /p:Platform=x64
# Build only the Open3D solution project:
msbuild OntheFlySfM.sln '/t:thirdparty\Open3DWeightedTSDF' /p:Configuration=Release /p:Platform=x64
~~~

The PowerShell script remains a shared implementation detail and can still be invoked directly. The generated Open3D projects are isolated from the framework's Directory.Build.props and vcpkg manifest settings.

Validation (2026-11-25): the modified Open3D library has been built from source and installed successfully with VS 2022 / CUDA 11.8. The build was performed in a new build directory, reusing only third-party source download archives. The helper supports an explicit VisualStudioInstance and ThirdPartyDownloadDirectory for restricted or offline environments.

The original build used shared libraries, the dynamic MSVC runtime, CUDA, and disabled GUI, Python bindings, ISPC, WebRTC, examples and tests. Only Release/x64 is the documented binary distribution. Install bin/Open3D.dll, lib/Open3D.lib, include/ and associated runtime dependencies together, preserving Open3D and bundled third-party notices.

For long-term maintenance, move this snapshot into a dedicated organization fork pinned to a commit and replace source/ with a Git submodule. Build helpers belong under scripts; modified library source belongs under thirdparty.
