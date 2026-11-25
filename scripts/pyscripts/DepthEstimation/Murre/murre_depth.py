import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
DEPTH_ROOT = SCRIPT_DIR.parent
if str(DEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ROOT))

from gpu_diagnostics import append_depth_gpu_memory, append_depth_quality, append_depth_worker_timing, timing_enabled
from output_resize_util import prepare_cropped_depth_output

MurrePipeline = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


def load_murre_modules():
    global MurrePipeline
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if MurrePipeline is not None:
        return

    try:
        from pipeline import MurrePipeline as pipeline_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "Murre dependencies are not importable. Required modules include "
            "pipeline.py, colmap_util.py, and the util/ package used by "
            f"pipeline.py. Original import error: {exc}"
        ) from exc

    MurrePipeline = pipeline_cls
    get_extrinsic = get_extrinsic_func
    get_hws = get_hws_func
    get_intrinsics = get_intrinsics_func
    read_model = read_model_func
    read_model_single_image = read_model_single_image_func


def qvec2rotmat(qvec):
    q0, q1, q2, q3 = qvec
    return np.array(
        [
            [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2],
        ],
        dtype=np.float64,
    )


def get_rescale_crop_tgthw(original_res, processing_res):
    h, w = original_res
    scale = min(processing_res / w, processing_res / h) if processing_res > 0 else 1.0
    new_h = int(h * scale)
    new_w = int(w * scale)
    crop_h = new_h - new_h % 16
    crop_w = new_w - new_w % 16
    return scale, crop_h, crop_w, new_h, new_w


def matrix_from_request(value, rows, cols, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (rows, cols):
        raise ValueError(f"Invalid {name} matrix shape: {arr.shape}, expected {(rows, cols)}")
    return arr


def metadata_from_request(request):
    request = request or {}
    image_file_name = request.get("image_file_name") or ""
    if not image_file_name or "camera_width" not in request or "camera_height" not in request:
        return None
    return {
        "image_file_name": image_file_name,
        "image_name": request.get("image_name") or os.path.splitext(image_file_name)[0],
        "original_res": (int(request["camera_height"]), int(request["camera_width"])),
        "intrinsic": matrix_from_request(request.get("intrinsic"), 3, 3, "intrinsic"),
        "pose": matrix_from_request(request.get("pose"), 4, 4, "pose"),
    }


def load_cpp_sparse_depth(request, crop_h, crop_w):
    sparse_depth_path = (request or {}).get("sparse_depth_path") or ""
    if not sparse_depth_path or not os.path.isfile(sparse_depth_path):
        return None
    sparse_depth = np.load(sparse_depth_path).astype(np.float32)
    if sparse_depth.shape != (crop_h, crop_w):
        raise ValueError(
            "C++ sparse depth shape mismatch for Murre: "
            f"{sparse_depth.shape}, expected {(crop_h, crop_w)}. Path: {sparse_depth_path}"
        )
    return sparse_depth


def generate_sparse_depth(ixt, ext, point3D_ids, points3D, h, w, max_depth=10.0):
    valid = [point_id for point_id in point3D_ids if point_id != -1 and point_id in points3D]
    if not valid:
        return np.zeros((h, w), dtype=np.float32)

    xyzs = np.array([points3D[point_id].xyz for point_id in valid])
    errors = np.array([points3D[point_id].error for point_id in valid])
    num_views = np.array([len(points3D[point_id].image_ids) for point_id in valid])

    cam_pts = xyzs @ ext[:3, :3].T + ext[:3, 3:].T
    pix = cam_pts @ ixt.T
    pix[:, :2] /= pix[:, 2:]

    depth_map = np.zeros((h, w), dtype=np.float32)
    for x, y, z, error, views in zip(pix[:, 0], pix[:, 1], cam_pts[:, 2], errors, num_views):
        px, py = int(x), int(y)
        if 0 <= px < w and 0 <= py < h and z > 0 and error < 10.0 and views > 1:
            depth_map[py, px] = min(float(z), max_depth)
    return depth_map


class MurreDepthGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        processing_res: int = 784,
        output_res: int = 784,
        max_depth: float = 10.0,
        denoise_steps: int = 4,
        ensemble_size: int = 1,
        color_map: str = "Spectral",
        device_str: str = "cuda",
    ):
        self.processing_res = processing_res
        self.output_res = output_res
        self.max_depth = max_depth
        self.denoise_steps = denoise_steps
        self.ensemble_size = ensemble_size
        self.color_map = color_map

        import torch

        self.torch = torch
        self.device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        load_murre_modules()
        self.pipe = MurrePipeline.from_pretrained(checkpoint_path, torch_dtype=self.dtype).to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def generate_depth(self, image_id, sfm_dir, image_folder, output_dir, request=None):
        request = request or {}
        metadata = metadata_from_request(request)
        points = None
        point3D_ids = None
        if metadata is not None:
            image_file_name = metadata["image_file_name"]
            image_name = metadata["image_name"]
            original_res = metadata["original_res"]
            ixt = metadata["intrinsic"]
            ext = metadata["pose"]
            original_ixt = ixt.copy()
        else:
            cams, image, points = read_model_single_image(sfm_dir, int(image_id))
            img_obj = image
            cam_obj = cams[img_obj.camera_id]
            image_file_name = img_obj.name
            image_name = os.path.splitext(img_obj.name)[0]
            original_res = get_hws(cam_obj)
            ixt = get_intrinsics(cam_obj)
            ext = get_extrinsic(img_obj)
            original_ixt = ixt.copy()
            point3D_ids = img_obj.point3D_ids

        img_path = os.path.join(image_folder, image_file_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        scale, crop_h, crop_w, tgt_h, tgt_w = get_rescale_crop_tgthw(original_res, self.processing_res)
        sparse_depth = load_cpp_sparse_depth(request, crop_h, crop_w)
        if sparse_depth is None:
            if point3D_ids is None or points is None:
                cams, image, points = read_model_single_image(sfm_dir, int(image_id))
                img_obj = image
                cam_obj = cams[img_obj.camera_id]
                image_file_name = img_obj.name
                image_name = os.path.splitext(img_obj.name)[0]
                original_res = get_hws(cam_obj)
                ixt = get_intrinsics(cam_obj)
                ext = get_extrinsic(img_obj)
                original_ixt = ixt.copy()
                point3D_ids = img_obj.point3D_ids
                scale, crop_h, crop_w, tgt_h, tgt_w = get_rescale_crop_tgthw(
                    original_res,
                    self.processing_res,
                )
            ixt = ixt.copy()
            ixt[:2] *= scale
            sparse_depth = generate_sparse_depth(ixt, ext, point3D_ids, points, tgt_h, tgt_w, self.max_depth)
            sparse_depth = sparse_depth[:crop_h, :crop_w]

        image_rgb = Image.open(img_path).convert("RGB")
        image_rgb = image_rgb.resize((tgt_w, tgt_h)).crop((0, 0, crop_w, crop_h))

        with self.torch.no_grad():
            result = self.pipe(
                image_rgb,
                sparse_depth,
                max_depth=self.max_depth,
                denoising_steps=self.denoise_steps,
                ensemble_size=self.ensemble_size,
                processing_res=self.processing_res,
                model_dtype=self.dtype,
                color_map=self.color_map,
                batch_size=1,
                show_progress_bar=False,
                resample_method="bilinear",
            )

        depth_pred = result.depth_np
        depth_colored = result.depth_colored
        original_h, original_w = original_res
        depth_pred, output_ixt, output_w, output_h, valid_w, valid_h = prepare_cropped_depth_output(
            depth_pred,
            original_ixt,
            original_w,
            original_h,
            self.output_res,
            16,
            cv2.INTER_LINEAR,
        )
        valid_color = depth_colored.resize((valid_w, valid_h), Image.BILINEAR)
        depth_colored = Image.new("RGB", (output_w, output_h), (0, 0, 0))
        depth_colored.paste(valid_color, (0, 0))

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

        np.save(depth_path, depth_pred)
        depth_colored.save(color_path)
        np.savetxt(intrinsic_path, output_ixt)
        np.savetxt(pose_path, ext)

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

    def test(self, image_id, sfm_dir, image_folder, output_dir):
        return {
            "depth_path": "",
            "color_path": "",
            "intrinsic_path": "",
            "pose_path": "",
            "height": 0,
            "width": 0,
        }


def write_json_line(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_predict(generator, request):
    image_id = int(request["image_id"])
    output_timing = timing_enabled(request)
    gpu_timing = timing_enabled(request.get("gpu_diagnostics", False))
    append_depth_gpu_memory(
        request["output_dir"], "Murre", "before_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    total_start = time.perf_counter()
    result = generator.generate_depth(
        image_id=image_id,
        sfm_dir=request["sfm_dir"],
        image_folder=request["image_folder"],
        output_dir=request["output_dir"],
        request=request,
    )
    total_ms = (time.perf_counter() - total_start) * 1000.0
    append_depth_gpu_memory(
        request["output_dir"], "Murre", "after_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    if output_timing:
        image_name = request.get("image_name", "")
        depth = np.load(result["depth_path"])
        append_depth_worker_timing(
            request["output_dir"], "Murre", image_id, image_name,
            generator.processing_res, result["width"], result["height"],
            {"total_ms": f"{total_ms:.3f}"},
            generator.torch, generator.device, output_timing)
        append_depth_quality(
            request["output_dir"], "Murre", image_id, image_name,
            "raw", depth, output_timing)
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
            generator = MurreDepthGenerator(
                checkpoint_path=args.checkpoint,
                processing_res=args.processing_res,
                output_res=args.output_res,
                max_depth=args.max_depth,
                denoise_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                device_str=args.device,
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
    generator = MurreDepthGenerator(
        checkpoint_path=args.checkpoint,
        processing_res=args.processing_res,
        output_res=args.output_res,
        max_depth=args.max_depth,
        denoise_steps=args.denoise_steps,
        ensemble_size=args.ensemble_size,
        device_str=args.device,
    )
    result = generator.generate_depth(args.image_id, args.sfm_dir, args.image_folder, args.output_dir)
    write_json_line({"ok": True, "image_id": args.image_id, **result})
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Murre depth prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="Murre checkpoint directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=784)
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)

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
