# C++ dependencies and local version inventory

Target: Windows x64, VS 2022, CUDA Toolkit **11.8 only**. Python is provided as a prebuilt environment by the project maintainer, not installed by vcpkg. Qt and modified Open3D are separate dependencies.

The root vcpkg.json was derived from source includes, project files and the local x64-windows installed status on 2026-11-24. The registry baseline is 7066011e27c97dbcfff739be2b69b592a41fc74a. Direct dependency versions and the Boost family are overridden to observed local versions. The complete installed inventory is in verified-vcpkg-packages.json; it includes unrelated packages and is not an exact reproducibility lockfile.

| Dependency | Observed version | Purpose |
| --- | --- | --- |
| Boost modules | 1.82.0#1 | algorithm, assert, config, filesystem, format, property-tree; transitive modules retained at a consistent Boost version |
| CGAL | 6.0.1 | triangulation, surface meshes and polygon mesh processing |
| Ceres | 2.2.0#2 | bundle adjustment; requested cuda/eigensparse/lapack/suitesparse/tools features |
| Eigen | 3.4.0#2 | linear algebra |
| Faiss | 1.7.4 | CPU HNSW retrieval |
| FLANN | 2019-04-07#7 | nearest-neighbor matching |
| FreeImage | 3.18.0#25 | image loading |
| glog | 0.7.1 | logging |
| GLEW | 2.2.0#2 | OpenGL extension loading |
| SQLite3 | 3.40.1#3 | Base/SQLite3Base.h directly includes sqlite3.h; project database |
| TBB | 2021.9.0 | concurrency |

GMP 6.3.0#2, MPFR 4.2.2, gflags 2.2.2#9, BLAS/LAPACK, SuiteSparse, LZ4 and image codecs are resolved transitively. Their exact resolved versions may differ from the installed inventory. SIFTGPU and VLFeat are built from bundled source. Modified Open3D is not the stock vcpkg open3d port. Do not install a stock Open3D package in its place.

## Known differences from the installed machine

- Installed Faiss reports a gpu feature, while the code uses CPU IndexHNSWFlat. The manifest intentionally does not request gpu because the registry port disallows it on Windows.
- Installed Ceres reports cxsparse, which is absent from the 2.2.0#2 registry port. The manifest uses the supported SuiteSparse features instead; it does not claim to recreate this inconsistent installed feature record.
- The vcpkg cuda port version is packaging metadata, not the installed NVIDIA Toolkit version. Install NVIDIA Toolkit 11.8 separately.
- TBB is supplied by vcpkg; legacy oneAPI include paths/integration flags have been removed. This inventory does not establish ABI compatibility with arbitrary manually installed SDKs.

## Installation

Use a separate vcpkg checkout so the existing global installation is preserved. Check out the baseline above, bootstrap vcpkg, run vcpkg integrate install, then run vcpkg install --triplet x64-windows from this repository. An up-to-date vcpkg executable is recommended for manifest support. Do not reuse old globally installed LIB/DLL files with the new manifest installation accidentally.

Configure VCPKG_ROOT/Visual Studio integration for that checkout. Project manifest root is this repository; dependencies are installed under vcpkg_installed/x64-windows. QtMsBuild, Qt SDK, CUDA Toolkit 11.8, Python and the modified Open3D installation must also be configured.

Validation: all pinned direct versions and Boost override revisions were found in the local registry version records. Dry-run planning did not finish: the default tool download was unavailable; forcing local tools then triggered a vcpkg Visual Studio detection internal error (visualstudio.cpp:89). No full clean dependency installation/build is claimed.

## 2026-11-25 follow-up

The direct dependency audit also includes bundled SIFTGPU (GLEW) and VLFeat. Added sqlite3 3.40.1#3 from the installed status; it is used directly by src/Base/SQLite3Base.h. Standard C++/Windows/OpenGL system headers, Qt, CUDA, Python and bundled modified Open3D are not additional vcpkg ports. Image codecs, gflags/GMP/MPFR and solver libraries remain transitive dependencies.

The nine framework/bundled-library projects now inherit manifest mode from Directory.Build.props; Open3D stays isolated. vcpkg-configuration.json selects an overlay x64-windows triplet with VCPKG_BUILD_TYPE=release, so only Release dependencies are built. The solution/project configurations are Release|x64 only.

Latest dry-run uses isolated buildtrees/packages/install directories outside the source tree. Port version records, including SQLite3, were found; full resolution/installation is not verified because the current vcpkg executable fails in Visual Studio detection (initially visualstudio.cpp:89; with explicit architecture/ProgramFiles variables, it reports no complete VS instance). No clean-manifest build has been claimed.

## Completed manifest rebuild — 2026-11-25

The current source has now completed a manifest Release/x64 build and UI startup check. The root manifest additionally pins 25 transitive ports to observed installed versions. The local build installed 146 core packages with binary-cache restore disabled, using source/tool download caches and explicit MSVC/Fortran/NASM/TBB environment overlays. This supersedes the earlier dry-run failure as the latest local result; it does not certify an unadapted clean-machine build. See BUILD_VERIFICATION.md for scope, output, fixes and limitations.
