import argparse
import contextlib
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
DEPTH_ROOT = SCRIPT_DIR.parent
if str(DEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ROOT))

from gpu_diagnostics import (
    append_depth_gpu_memory,
    append_depth_quality,
    append_depth_worker_timing,
    timing_enabled,
)
from output_resize_util import prepare_cropped_depth_output

PriorDepthAnything = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


def load_modules():
    global PriorDepthAnything
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if PriorDepthAnything is not None:
        return

    try:
        from prior_depth_anything import PriorDepthAnything as priorda_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "PriorDepthAnything dependencies are not importable. Expected the "
            "prior_depth_anything package and colmap_util.py next to this worker. "
            f"Original import error: {exc}"
        ) from exc

    PriorDepthAnything = priorda_cls
    get_extrinsic = get_extrinsic_func
    get_hws = get_hws_func
    get_intrinsics = get_intrinsics_func
    read_model = read_model_func
    read_model_single_image = read_model_single_image_func


def get_rescale_crop_tgthw(original_res, processing_res):
    h, w = original_res
    if processing_res <= 0:
        return 1.0, h, w, h, w
    scale = min(processing_res / w, processing_res / h)
    new_h = int(h * scale)
    new_w = int(w * scale)
    crop_h = new_h - new_h % 14
    crop_w = new_w - new_w % 14
    return scale, crop_h, crop_w, new_h, new_w


def generate_sparse_depth_from_observations(
    ext,
    xys,
    point3d_ids,
    points3d,
    h,
    w,
    scale=1.0,
    max_depth=50.0,
):
    stats = {
        "num_observations": int(len(point3d_ids)),
        "num_valid_indices": 0,
        "num_valid_sparse_points": 0,
    }
    valid_indices = [
        idx for idx, point_id in enumerate(point3d_ids)
        if point_id != -1 and point_id in points3d
    ]
    stats["num_valid_indices"] = int(len(valid_indices))
    depth_map = np.zeros((h, w), dtype=np.float32)
    if not valid_indices:
        return depth_map, stats

    valid_point_ids = [int(point3d_ids[idx]) for idx in valid_indices]
    xyzs = np.array([points3d[point_id].xyz for point_id in valid_point_ids])
    obs_xys = np.asarray(xys[valid_indices], dtype=np.float32) * float(scale)
    errors = np.array([points3d[point_id].error for point_id in valid_point_ids])
    num_views = np.array([len(points3d[point_id].image_ids) for point_id in valid_point_ids])

    cam_pts = xyzs @ ext[:3, :3].T + ext[:3, 3:].T

    for x, y, z, error, views in zip(obs_xys[:, 0], obs_xys[:, 1], cam_pts[:, 2], errors, num_views):
        px, py = int(round(float(x))), int(round(float(y)))
        if 0 <= px < w and 0 <= py < h and z > 0 and error < 10.0 and views > 1:
            z = min(float(z), max_depth)
            if depth_map[py, px] <= 0.0 or z < depth_map[py, px]:
                depth_map[py, px] = z
    stats["num_valid_sparse_points"] = int(np.count_nonzero(depth_map > 0.0))
    return depth_map, stats


def elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def dense_reconstruction_dir_from_depth_dir(output_dir):
    return os.path.dirname(os.path.normpath(output_dir))


def sync_cuda(torch_module, device):
    if str(device).lower().startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def cuda_memory_stats(torch_module, device):
    if not str(device).lower().startswith("cuda") or not torch_module.cuda.is_available():
        return {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_max_allocated_mb": 0.0,
        }
    return {
        "cuda_allocated_mb": torch_module.cuda.memory_allocated() / (1024.0 * 1024.0),
        "cuda_reserved_mb": torch_module.cuda.memory_reserved() / (1024.0 * 1024.0),
        "cuda_max_allocated_mb": torch_module.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    }


def append_timing_row(output_dir, row, enabled=True):
    if not timing_enabled(enabled):
        return
    timing_dir = os.path.join(dense_reconstruction_dir_from_depth_dir(output_dir), "Timing")
    os.makedirs(timing_dir, exist_ok=True)
    timing_path = os.path.join(timing_dir, "priorda_worker_timing.csv")
    fieldnames = [
        "timestamp",
        "image_id",
        "image_name",
        "device",
        "depth_width",
        "depth_height",
        "num_points3d_total",
        "num_observations",
        "num_valid_indices",
        "num_valid_sparse_points",
        "sparse_depth_source",
        "read_sparse_depth_ms",
        "read_model_ms",
        "sparse_prior_ms",
        "image_preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "write_outputs_ms",
        "total_ms",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_max_allocated_mb",
    ]
    write_header = not os.path.isfile(timing_path) or os.path.getsize(timing_path) == 0
    with open(timing_path, "a", newline="", encoding="utf-8") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fid.flush()


def colorize_depth(depth, max_depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0)
    if valid.any():
        lo = float(np.percentile(depth[valid], 2))
        hi = float(np.percentile(depth[valid], 98))
    else:
        lo, hi = 0.0, float(max_depth)
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    try:
        from matplotlib import colormaps
        colored = colormaps["Spectral"](normalized)[..., :3]
        colored[~valid] = 0.0
        return Image.fromarray((colored * 255.0).astype(np.uint8)).convert("RGB")
    except Exception:
        gray = (normalized * 255.0).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")


def matrix_from_request(value, rows, cols, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, cols):
        raise ValueError(f"{name} must have shape {rows}x{cols}, got {array.shape}")
    return array


def load_cpp_sparse_depth(request, crop_h, crop_w):
    sparse_depth_path = request.get("sparse_depth_path") or ""
    if not sparse_depth_path or not os.path.isfile(sparse_depth_path):
        return None, 0.0

    section_start = time.perf_counter()
    sparse_depth = np.load(sparse_depth_path).astype(np.float32)
    read_sparse_depth_ms = elapsed_ms(section_start)
    if sparse_depth.shape != (crop_h, crop_w):
        print(
            "[PriorDA] Ignoring C++ sparse depth with mismatched shape: "
            f"{sparse_depth.shape}, expected {(crop_h, crop_w)}. Path: {sparse_depth_path}",
            file=sys.stderr,
        )
        return None, read_sparse_depth_ms
    sparse_depth[~np.isfinite(sparse_depth)] = 0.0
    sparse_depth[sparse_depth <= 0.0] = 0.0
    return sparse_depth, read_sparse_depth_ms


class PriorDepthAnythingGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        processing_res: int = 784,
        output_res: int = 784,
        max_depth: float = 50.0,
        device_str: str = "cuda",
        version: str = "1.1",
        frozen_model_size: str = "vitb",
        conditioned_model_size: str = "vitb",
        pattern: str = None,
    ):
        import torch

        load_modules()
        self.torch = torch
        if processing_res > 0 and processing_res % 14 != 0:
            raise ValueError("PriorDAProcessingResolution must be <= 0 or a multiple of 14.")
        self.processing_res = processing_res
        self.output_res = output_res
        self.max_depth = max_depth
        self.pattern = pattern

        use_cuda = device_str.lower().startswith("cuda") and torch.cuda.is_available()
        self.device = "cuda:0" if use_cuda else "cpu"
        self.model = PriorDepthAnything(
            device=self.device,
            version=version,
            mde_dir=checkpoint_path,
            ckpt_dir=checkpoint_path,
            frozen_model_size=frozen_model_size,
            conditioned_model_size=conditioned_model_size,
        ).eval()

    def generate_depth(
        self,
        image_id,
        sfm_dir,
        image_folder,
        output_dir,
        request=None,
    ):
        request = request or {}
        total_start = time.perf_counter()
        read_model_ms = 0.0
        read_sparse_depth_ms = 0.0
        sparse_prior_ms = 0.0
        image_preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0
        write_outputs_ms = 0.0
        num_points3d_total = 0
        sparse_depth_source = "sfm_fallback"
        sparse_stats = {
            "num_observations": 0,
            "num_valid_indices": 0,
            "num_valid_sparse_points": 0,
        }

        image_file_name = request.get("image_file_name") or ""
        image_name = request.get("image_name") or (
            os.path.splitext(image_file_name)[0] if image_file_name else ""
        )
        ixt = None
        ext = None
        original_res = None
        using_request_metadata = False
        if image_file_name and "camera_width" in request and "camera_height" in request:
            original_res = (int(request["camera_height"]), int(request["camera_width"]))
            ixt = matrix_from_request(request.get("intrinsic"), 3, 3, "intrinsic")
            ext = matrix_from_request(request.get("pose"), 4, 4, "pose")
            using_request_metadata = True

        if original_res is None:
            section_start = time.perf_counter()
            cams, image, points = read_model_single_image(sfm_dir, int(image_id))
            read_model_ms = elapsed_ms(section_start)
            num_points3d_total = int(len(points))
            img_obj = image
            cam_obj = cams[img_obj.camera_id]
            image_file_name = img_obj.name
            image_name = os.path.splitext(img_obj.name)[0]
            original_res = get_hws(cam_obj)
            ixt = get_intrinsics(cam_obj)
            ext = get_extrinsic(img_obj)
        else:
            points = None
            img_obj = None
            read_model_ms = 0.0

        img_path = os.path.join(image_folder, image_file_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        scale, crop_h, crop_w, tgt_h, tgt_w = get_rescale_crop_tgthw(original_res, self.processing_res)
        original_ixt = ixt.copy()
        ixt = ixt.copy()
        if not using_request_metadata:
            ixt[:2] *= scale

        sparse_depth = None
        if request.get("sparse_depth_path"):
            sparse_depth, read_sparse_depth_ms = load_cpp_sparse_depth(request, crop_h, crop_w)
            if sparse_depth is not None:
                sparse_depth_source = "cpp_sparse_depth"
                sparse_stats = {
                    "num_observations": int(request.get("num_observations", 0)),
                    "num_valid_indices": int(request.get("num_valid_indices", 0)),
                    "num_valid_sparse_points": int(np.count_nonzero(sparse_depth > 0.0)),
                }

        if sparse_depth is None:
            if img_obj is None or points is None:
                section_start = time.perf_counter()
                cams, image, points = read_model_single_image(sfm_dir, int(image_id))
                read_model_ms = elapsed_ms(section_start)
                num_points3d_total = int(len(points))
                img_obj = image
                cam_obj = cams[img_obj.camera_id]
                image_file_name = img_obj.name
                image_name = os.path.splitext(img_obj.name)[0]
                original_res = get_hws(cam_obj)
                ixt = get_intrinsics(cam_obj)
                ext = get_extrinsic(img_obj)
                using_request_metadata = False
                scale, crop_h, crop_w, tgt_h, tgt_w = get_rescale_crop_tgthw(
                    original_res,
                    self.processing_res,
                )
                original_ixt = ixt.copy()
                ixt = ixt.copy()
                ixt[:2] *= scale

            section_start = time.perf_counter()
            sparse_depth, sparse_stats = generate_sparse_depth_from_observations(
                ext,
                img_obj.xys,
                img_obj.point3D_ids,
                points,
                crop_h,
                crop_w,
                scale,
                self.max_depth,
            )
            sparse_prior_ms = elapsed_ms(section_start)

        section_start = time.perf_counter()
        image_rgb = Image.open(img_path).convert("RGB")
        if self.processing_res > 0:
            image_rgb = image_rgb.resize((tgt_w, tgt_h)).crop((0, 0, crop_w, crop_h))
        image_np = np.asarray(image_rgb, dtype=np.uint8)
        image_preprocess_ms = elapsed_ms(section_start)

        sync_cuda(self.torch, self.device)
        section_start = time.perf_counter()
        with self.torch.no_grad():
            depth = self.model.infer_one_sample(
                image=image_np,
                prior=sparse_depth,
                geometric=None,
                pattern=self.pattern,
                visualize=False,
            )
        sync_cuda(self.torch, self.device)
        inference_ms = elapsed_ms(section_start)

        section_start = time.perf_counter()
        depth_pred = depth.detach().cpu().numpy().astype(np.float32)
        depth_pred[~np.isfinite(depth_pred)] = 0.0
        depth_pred = np.clip(depth_pred, 0.0, self.max_depth)
        original_h, original_w = original_res
        depth_pred, output_ixt, _output_w, _output_h, _valid_w, _valid_h = prepare_cropped_depth_output(
            depth_pred,
            original_ixt,
            original_w,
            original_h,
            self.output_res,
            14,
            cv2.INTER_LINEAR,
        )
        postprocess_ms = elapsed_ms(section_start)

        np_output = os.path.join(output_dir, "np")
        color_output = os.path.join(output_dir, "color")
        intrinsic_output = os.path.join(output_dir, "intrinsic")
        pose_output = os.path.join(output_dir, "pose")

        os.makedirs(np_output, exist_ok=True)
        os.makedirs(color_output, exist_ok=True)
        os.makedirs(intrinsic_output, exist_ok=True)
        os.makedirs(pose_output, exist_ok=True)

        depth_path = os.path.join(np_output, f"{image_name}_depth.npy")
        color_path = os.path.join(color_output, f"{image_name}_color.png")
        intrinsic_path = os.path.join(intrinsic_output, f"{image_name}.txt")
        pose_path = os.path.join(pose_output, f"{image_name}.txt")

        section_start = time.perf_counter()
        np.save(depth_path, depth_pred)
        colorize_depth(depth_pred, self.max_depth).save(color_path)
        np.savetxt(intrinsic_path, output_ixt)
        np.savetxt(pose_path, ext)
        write_outputs_ms = elapsed_ms(section_start)

        timing_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_id": int(image_id),
            "image_name": image_name,
            "device": self.device,
            "depth_width": int(depth_pred.shape[1]),
            "depth_height": int(depth_pred.shape[0]),
            "num_points3d_total": num_points3d_total,
            "num_observations": int(sparse_stats["num_observations"]),
            "num_valid_indices": int(sparse_stats["num_valid_indices"]),
            "num_valid_sparse_points": int(sparse_stats["num_valid_sparse_points"]),
            "sparse_depth_source": sparse_depth_source,
            "read_sparse_depth_ms": f"{read_sparse_depth_ms:.3f}",
            "read_model_ms": f"{read_model_ms:.3f}",
            "sparse_prior_ms": f"{sparse_prior_ms:.3f}",
            "image_preprocess_ms": f"{image_preprocess_ms:.3f}",
            "inference_ms": f"{inference_ms:.3f}",
            "postprocess_ms": f"{postprocess_ms:.3f}",
            "write_outputs_ms": f"{write_outputs_ms:.3f}",
            "total_ms": f"{elapsed_ms(total_start):.3f}",
            **{
                key: f"{value:.3f}"
                for key, value in cuda_memory_stats(self.torch, self.device).items()
            },
        }
        output_timing = timing_enabled(request)
        try:
            append_timing_row(output_dir, timing_row, output_timing)
            append_depth_worker_timing(
                output_dir,
                "PriorDepthAnything",
                int(image_id),
                image_name,
                self.processing_res,
                int(depth_pred.shape[1]),
                int(depth_pred.shape[0]),
                {
                    "preprocess_ms": f"{image_preprocess_ms:.3f}",
                    "inference_ms": f"{inference_ms:.3f}",
                    "postprocess_ms": f"{postprocess_ms:.3f}",
                    "write_outputs_ms": f"{write_outputs_ms:.3f}",
                    "total_ms": f"{elapsed_ms(total_start):.3f}",
                },
                self.torch,
                self.device,
                output_timing,
            )
            append_depth_quality(
                output_dir,
                "PriorDepthAnything",
                int(image_id),
                image_name,
                "raw",
                depth_pred,
                output_timing,
            )
        except Exception as exc:
            print(f"[PriorDA Timing] Failed to write timing row: {exc}", file=sys.stderr)

        return {
            "depth_path": depth_path,
            "color_path": color_path,
            "intrinsic_path": intrinsic_path,
            "pose_path": pose_path,
            "height": int(depth_pred.shape[0]),
            "width": int(depth_pred.shape[1]),
        }

    def generate_depth_from_name(self, image_name: str, sfm_dir: str, image_folder: str, output_dir: str):
        _cams, images, _points = read_model(sfm_dir)
        matched = [image_id for image_id, image in images.items() if image.name == image_name]
        if not matched:
            raise ValueError(f"Image {image_name} is not found in COLMAP model: {sfm_dir}")
        return self.generate_depth(matched[0], sfm_dir, image_folder, output_dir)


def write_json_line(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_predict(generator, request):
    image_id = int(request["image_id"])
    output_timing = timing_enabled(request)
    gpu_timing = timing_enabled(request.get("gpu_diagnostics", False))
    append_depth_gpu_memory(
        request["output_dir"], "PriorDepthAnything", "before_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    result = generator.generate_depth(
        image_id=image_id,
        sfm_dir=request["sfm_dir"],
        image_folder=request["image_folder"],
        output_dir=request["output_dir"],
        request=request,
    )
    append_depth_gpu_memory(
        request["output_dir"], "PriorDepthAnything", "after_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    return {
        "type": "result",
        "request_id": request.get("request_id"),
        "image_id": image_id,
        "ok": True,
        **result,
    }


def run_worker(args):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            generator = PriorDepthAnythingGenerator(
                checkpoint_path=args.checkpoint,
                processing_res=args.processing_res,
                output_res=args.output_res,
                max_depth=args.max_depth,
                device_str=args.device,
                version=args.version,
                frozen_model_size=args.frozen_model_size,
                conditioned_model_size=args.conditioned_model_size,
                pattern=args.pattern,
            )
        write_json_line({"type": "ready", "ok": True})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        write_json_line({"type": "ready", "ok": False, "error": str(exc)})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            request_type = request.get("type", "predict")
            if request_type == "shutdown":
                write_json_line({"type": "shutdown", "ok": True})
                return 0
            if request_type != "predict":
                raise ValueError(f"Unsupported request type: {request_type}")

            with contextlib.redirect_stdout(sys.stderr):
                response = handle_predict(generator, request)
            write_json_line(response)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write_json_line(
                {
                    "type": "result",
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "image_id": request.get("image_id") if "request" in locals() else None,
                    "ok": False,
                    "error": str(exc),
                }
            )

    return 0


def run_single(args):
    generator = PriorDepthAnythingGenerator(
        checkpoint_path=args.checkpoint,
        processing_res=args.processing_res,
        output_res=args.output_res,
        max_depth=args.max_depth,
        device_str=args.device,
        version=args.version,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        pattern=args.pattern,
    )
    result = generator.generate_depth(args.image_id, args.sfm_dir, args.image_folder, args.output_dir)
    write_json_line({"ok": True, "image_id": args.image_id, **result})
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="PriorDepthAnything depth prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="PriorDepthAnything checkpoint directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=784)
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--version", default="1.1", choices=["1.0", "1.1"])
    parser.add_argument("--frozen-model-size", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--conditioned-model-size", default="vitb", choices=["vits", "vitb"])
    parser.add_argument("--pattern", default=None)

    parser.add_argument("--image-id", type=int, help="Single-shot COLMAP image id.")
    parser.add_argument("--sfm-dir", help="Single-shot COLMAP sparse model directory.")
    parser.add_argument("--image-folder", help="Single-shot image folder.")
    parser.add_argument("--output-dir", help="Single-shot output directory.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)

    missing = [
        name
        for name in ("image_id", "sfm_dir", "image_folder", "output_dir")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("single-shot mode requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
