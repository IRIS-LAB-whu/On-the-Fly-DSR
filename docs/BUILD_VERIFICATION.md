# Build verification — November 25, 2026

A full Windows x64 / Release / CUDA 11.8 build completed after old outputs were removed. Modified Open3D was compiled from source, the manifest installation contained 146 core packages, and the complete solution build exited successfully. Binary-cache restore was disabled; cached source archives and tools were reused.

The deployed UI started and responded. Logs confirmed SupScene loading and DepthAnythingV2 import. Runtime dependency checks resolved 62 non-system dependencies; 71 DLLs were present at the application root. Dataset reconstruction was not repeated in this build. The maintainer previously reported successful phantom3 reconstruction.

## Scope and limitations

The local environment needed explicit MSVC/Windows SDK configuration and local overlays for GNU Fortran/LAPACK import libraries and runtimes, cached NASM, and TBB HWLOC detection. Those machine-specific adaptations are not part of the release source. This verifies that adapted local build, not the unmodified instructions on a fresh machine.

Environment: VS 2022 Community, MSVC 14.38.33130, Qt 5.15.2, CUDA 11.8.89, Python 3.10.10. Resolved packages are recorded in [verified-vcpkg-packages.json](verified-vcpkg-packages.json). Compiler warnings remain.

After this verification, publication cleanup removed caches and unused files and replaced machine-specific default paths with portable examples. Those final path/documentation changes were not followed by another complete build.
