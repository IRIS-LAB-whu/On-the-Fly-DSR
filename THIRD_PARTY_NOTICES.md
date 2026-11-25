# Licensing scope and third-party notices

The root LICENSE contains the unmodified GNU GPL version 3 text. The intended license for project-owned original contributions is GPL-3.0-or-later; see NOTICE. This does not grant permissions over copied third-party code or model weights and is not a certification that the complete current source/binary distribution is license-compatible.

## Release issues requiring resolution

1. Bundled SiftGPU headers grant educational, research and non-profit use, copying, modification and distribution. This is not an unrestricted open-source license. Preserve thirdparty/SIFTGPU/LICENSE and all file headers. Resolve redistribution/combination permissions or replace/isolate this dependency before claiming the entire framework is GPL-compatible or suitable for unrestricted commercial use.
2. CGAL 3D triangulations and Polygon Mesh Processing use GPL terms (or a separate commercial license). The GPL choice for original framework contributions reflects these dependencies; MIT/Apache on the whole combined program would not by itself remove those obligations. See https://www.cgal.org/license.html and the exact installed package's license files.
3. src/Scene/graph.h references an upstream README.TXT for its MaxFlow license, but that original README is missing here. Identify the exact upstream release and restore its license; do not substitute an arbitrary modern MaxFlow license.
4. The original On-the-Fly SfM repository publishes BSD-3-Clause, copyright (c) 2025 IRIS-LAB whu. Its exact license has been retained in licenses/upstream/On-the-Fly-SfM.LICENSE (source: https://github.com/sygant/onthefly/blob/main/LICENSE; retrieved 2026-11-25; Git blob SHA 751d0d0397d5b4ba71681bfbc4d8896ccf52d7ef). Preserve this attribution and file-level notices; it does not license SfM's dependencies or establish the provenance of every copied file.
5. Prebuilt Python/Open3D archives, DLLs and model checkpoints must carry their own dependency notices and any applicable source/license obligations, even when distributed through Google Drive.

## Included notices and official reference downloads

| Component | Location / source | Status |
| --- | --- | --- |
| Original On-the-Fly SfM | licenses/upstream/On-the-Fly-SfM.LICENSE | BSD-3-Clause, IRIS-LAB whu, 2025 |
| Modified Open3D 0.18.0 | thirdparty/Open3D-weightedTSDF/LICENSE and source/LICENSE | MIT; copied from the exact local source snapshot; embedded dependencies keep their own notices |
| VLFeat 0.9.20 | thirdparty/VLFeat/COPYING | Official v0.9.20 COPYING; local generic.h declares 0.9.20 |
| SiftGPU | thirdparty/SIFTGPU/LICENSE | Extracted from the bundled source's permission notice; restricted purposes |
| Depth Anything V2 | scripts/pyscripts/DepthEstimation/DepthAnythingV2/LICENSE | Official Apache-2.0 source license reference; weights have separate terms |
| DINOv2 bundled inside SupScene | scripts/pyscripts/Supscene/third_party/dinov2/LICENSE | Preserve existing upstream file; internal backbone is still required |
| UI assets | src/UI/media/LICENSE | Preserve existing asset license |
| Other model licenses | licenses/upstream/ and licenses/upstream-sources.json | Current official reference texts, not proof of the copied source version's license |

Downloaded reference files for Murre, MoGe, Prior Depth Anything, Pixel-Perfect Depth and Depth Anything 3 are stored under licenses/upstream, rather than silently relabeling local copied model trees. No original source commit metadata was available for those local copies. Match each copied source/checkpoint to its upstream revision and retain all applicable LICENSE/NOTICE files before release. In particular, the current Murre license is PRL, so do not assume a current license matches an older local copy.

Code licenses and checkpoint licenses differ. Depth Anything V2 Base/Large/Giant model weights have non-commercial terms even though the repository code is Apache-2.0. DA3 and nested model backbones/checkpoints also need individual model-card review. Downloading a LICENSE file alone does not authorize redistribution or use of the weights. Prefer official model download links and record model ID, revision, license and checksum.

## Official depth-model references — 2026-11-25

All six current official LICENSE files are stored under licenses/upstream; see licenses/upstream-sources.json for URLs and Git blob SHAs. PixelPerfectDepth uses the maintainer-supplied gangweix/pixel-perfect-depth source URL. Preserve nested notices in these files (including MoGe DINOv2 terms). These references do not establish the source revision of the local copies. See licenses/README.md for source/binary/download distribution guidance.

SiftGPU and MaxFlow follow-up is deferred at the maintainer's request; existing notices and unresolved status are retained.
