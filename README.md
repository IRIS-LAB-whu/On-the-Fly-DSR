# On-the-Fly DSR

<img src="OntheflyIcon.jpg" width="80" alt="On-the-Fly DSR">

An online sparse and dense scene reconstruction framework built on [On-the-Fly SfM](https://github.com/sygant/onthefly). It combines incremental camera pose estimation, monocular depth prediction, SfM-guided depth alignment, and weighted TSDF fusion using a modified Open3D implementation.

## Pipeline

![On-the-Fly DSR pipeline](docs/media/main_pipeline.svg)

https://github.com/user-attachments/assets/a0b42c40-42d8-4a89-98d9-d848906dbdc4


## Requirements

The supported configuration is **Windows x64 / Release / CUDA 11.8**.

| Component | Requirement |
| --- | --- |
| Compiler | Visual Studio 2022, MSVC v143, Windows SDK, Desktop development with C++ |
| Qt | **5.15.2**, MSVC 2019 64-bit SDK; Qt VS Tools / QtMsBuild |
| Qt modules | Core, Gui, Widgets, OpenGL, Concurrent, WinExtras, Qt3D, Charts |
| CUDA | NVIDIA CUDA Toolkit **11.8** and a compatible NVIDIA driver |
| CMake | 3.21 or newer |
| C++ packages | vcpkg manifest in `vcpkg.json` |
| Python | Maintainer-provided Python 3.10 environment |
| Open3D | Modified 0.18.0 source included in `thirdparty/Open3D-weightedTSDF/source` |

## Downloads

The Python environment and image retrieval checkpoints are distributed separately through Google Drive and are **not included in this repository**.

| Asset | Download | Destination |
| --- | --- | --- |
| Python environment | [Google Drive](https://drive.google.com/file/d/1ERBA1Y4TpmMwl-q55n_pRF2EQdsBwSoL/view?usp=drive_link) | Extract directly into `thirdparty/Python/` |
| SupScene: `dinov2_scpp_supscene_1536.pth` | [Checkpoints — Google Drive](https://drive.google.com/drive/folders/1dxRzWBmeBJaE6ZsoDhq-Kt_aBBscgz9G?usp=drive_link) | `scripts/pyscripts/Supscene/checkpoints/` |
| Optional VGG/PCA/NetVLAD weights | [Checkpoints — Google Drive](https://drive.google.com/drive/folders/1dxRzWBmeBJaE6ZsoDhq-Kt_aBBscgz9G?usp=drive_link) | `scripts/pyscripts/GlobalVGG/models/` |

The default pipeline uses **SupScene + DepthAnythingV2 (vitl)**. The Python directory must directly contain `python.exe`, `python310.dll`, `include`, and `libs`.

### Monocular depth estimation models

Download depth model weights from the original projects below. These weights are separate from the SupScene/VGG checkpoints linked above.

| Backend | Official repository |
| --- | --- |
| Murre | [zju3dv/Murre](https://github.com/zju3dv/Murre) |
| PriorDepthAnything | [SpatialVision/Prior-Depth-Anything](https://github.com/SpatialVision/Prior-Depth-Anything) |
| DepthAnythingV2 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| DepthAnything3 | [ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) |
| PixelPerfectDepth | [gangweix/pixel-perfect-depth](https://github.com/gangweix/pixel-perfect-depth) |
| MoGe | [microsoft/MoGe](https://github.com/microsoft/MoGe) |

For the default DepthAnythingV2 / vitl configuration, download `depth_anything_v2_vitl.pth` from the DepthAnythingV2 project and place it in `scripts/pyscripts/DepthEstimation/DepthAnythingV2/checkpoints/`.

Match the downloaded weights to the selected adapter and encoder. PixelPerfectDepth also requires the corresponding DA2 or MoGe2 backbone weights. Code and weights can have different licenses; consult each original project and [third-party notices](THIRD_PARTY_NOTICES.md).

## Build

Clone the repository:

~~~powershell
git clone https://github.com/IRIS-LAB-whu/On-the-Fly-DSR.git
cd On-the-Fly-DSR
~~~

Install/bootstrap vcpkg, add it to PATH, and run in a Visual Studio developer PowerShell:

~~~powershell
vcpkg integrate install
vcpkg install --triplet x64-windows
~~~

Manifest mode reads `vcpkg.json`, resolves direct and transitive dependencies against its baseline/version overrides, and installs them into this repository's `vcpkg_installed` directory. The overlay triplet builds Release packages only. See [dependency details](docs/DEPENDENCIES.md).

1. Extract the Python environment and download the required weights.
2. Register Qt 5.15.2 in Qt VS Tools, or configure `QTDIR` / MSBuild `QtSdkRoot`.
3. Open `OntheFlySfM.sln`, select **Release / x64**, and build the solution.
4. `Open3DWeightedTSDF` builds and installs the bundled modified Open3D first. Its first build needs network access for dependencies.
5. The UI deployment step copies runtime DLLs and Qt plugins and checks for missing dependencies.
6. Start `UI.exe` with its containing directory as the working directory.

Do not substitute a stock Open3D package for the bundled weighted TSDF implementation. Project dependencies are stored in the solution/project files; see [the project graph](docs/PROJECT_DEPENDENCIES.md).

A full local manifest/Release build and UI startup check completed on November 25, 2026. The verification used local toolchain adaptations; a fresh-machine build without those adaptations has not been verified. See [build verification](docs/BUILD_VERIFICATION.md).

## Configuration and first run

`src/settings.ini` is the single default configuration. On first deployment, the script creates `settings.ini` in the parent directory of `UI.exe` and fills in the local Python paths. Existing runtime settings are preserved. Edit that runtime file after building.

| Setting | What to configure |
| --- | --- |
| `[Python] PythonHome` | Absolute path to `thirdparty/Python` |
| `[Python] PythonWorkspace` | Absolute path to `scripts/pyscripts` |
| `[GlobalFeature] GlobalFeatureType` | SupScene by default; VGG needs its optional weights |
| `SupSceneWeightsPath` | `auto` uses the default checkpoints location; otherwise provide a file path |
| `[DenseDepth] Method / DepthAnythingV2Encoder` | Defaults to DepthAnythingV2 / vitl; match the weights |
| `[DepthAlignment] Enable` | Keep enabled when aligning relative depth for TSDF fusion |
| `[Pipeline] AutoRunImages` | Enable when processing an existing image directory |
| `[Experiment] Enable` | Disabled by default; enable for automatic example execution |
| `[Experiment] ProjectPath / ImagePath` | Database output path and input image directory for automatic execution |
| `[Test] ModelFolder` | Optional test functionality; leave empty for normal reconstruction |

Use absolute paths with forward slashes. For interactive use, create a project in the UI and select the images and output database.

## Sample data

[examples/phantom3](examples/phantom3/README.md) contains 50 original JPG images (approximately 51 MiB), without resizing or re-encoding. Reconstruction of the original phantom3 dataset was reported successful by the maintainer. The included subset has been checked for copy integrity; separate reconstruction metrics for that subset have not been recorded.

## License and acknowledgments

Original On-the-Fly SfM code retains its [BSD-3-Clause license](licenses/upstream/On-the-Fly-SfM.LICENSE). Project-owned additions are declared under GPL-3.0-or-later; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party code, models, and data retain their respective terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [license sources](licenses/upstream-sources.json). Outstanding SiftGPU/MaxFlow license questions and upstream-version checks are documented there; the repository does not claim that every third-party license has been cleared.

## Related publications

1. Zhan, Z., Yu, Y., Xia, R., Gan, W., Xie, H., Perda, G., Morelli, L., Remondino, F., and Wang, X. (2025). SfM on-the-fly: A robust near real-time SfM for spatiotemporally disordered high-resolution imagery from multiple agents. *ISPRS Journal of Photogrammetry and Remote Sensing*, **224**, 202–221. [doi:10.1016/j.isprsjprs.2025.04.002](https://doi.org/10.1016/j.isprsjprs.2025.04.002).

2. Dou, C., Yu, Y., Wang, X., and Zhan, Z. (2026). Learning-based Monocular Depth Estimation for Photogrammetric 3D Reconstruction. *The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, **XLIX-B2-2026**, 1345–1354. [doi:10.5194/isprs-archives-XLIX-B2-2026-1345-2026](https://doi.org/10.5194/isprs-archives-XLIX-B2-2026-1345-2026).
