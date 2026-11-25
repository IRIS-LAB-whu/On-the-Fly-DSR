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
from output_resize_util import prepare_depth_output

DepthAnythingV2 = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def load_modules():
    global DepthAnythingV2
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if DepthAnythingV2 is not None:
        return

    try:
        from depth_anything_v2.dpt import DepthAnythingV2 as depth_anything_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "DepthAnythingV2 dependencies are not importable. Expected the "
            "depth_anything_v2 package and colmap_util.py next to this worker. "
            f"Original import error: {exc}"
        ) from exc

    DepthAnythingV2 = depth_anything_cls
    get_extrinsic = get_extrinsic_func
    get_hws = get_hws_func
    get_intrinsics = get_intrinsics_func
    read_model = read_model_func
    read_model_single_image = read_model_single_image_func


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

        colored = colormaps["Spectral_r"](normalized)[..., :3]
        colored[~valid] = 0.0
        return Image.fromarray((colored * 255.0).astype(np.uint8)).convert("RGB")
    except Exception:
        gray = (normalized * 255.0).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")


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
        "original_h": int(request["camera_height"]),
        "original_w": int(request["camera_width"]),
        "intrinsic": matrix_from_request(request.get("intrinsic"), 3, 3, "intrinsic"),
        "pose": matrix_from_request(request.get("pose"), 4, 4, "pose"),
    }


class DepthAnythingV2Generator:
    def __init__(
        self,
        checkpoint_path: str,
        processing_res: int = 784,
        output_res: int = 784,
        max_depth: float = 50.0,
        device_str: str = "cuda",
        encoder: str = "vitb",
    ):
        import torch

        load_modules()
        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported DepthAnythingV2 encoder: {encoder}")
        if processing_res <= 0 or processing_res % 14 != 0:
            raise ValueError("DepthAnythingV2ProcessingResolution must be a positive multiple of 14.")

        self.torch = torch
        self.processing_res = processing_res
        self.output_res = output_res
        self.max_depth = max_depth
        self.encoder = encoder

        use_cuda = device_str.lower().startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        ckpt_file = os.path.join(checkpoint_path, f"depth_anything_v2_{encoder}.pth")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(f"DepthAnythingV2 checkpoint not found: {ckpt_file}")

        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    def generate_depth(self, image_id, sfm_dir, image_folder, output_dir, request=None):
        metadata = metadata_from_request(request)
        if metadata is not None:
            image_file_name = metadata["image_file_name"]
            image_name = metadata["image_name"]
            original_h = metadata["original_h"]
            original_w = metadata["original_w"]
            ixt = metadata["intrinsic"]
            ext = metadata["pose"]
        else:
            cams, image, _points = read_model_single_image(sfm_dir, int(image_id))
            img_obj = image
            cam_obj = cams[img_obj.camera_id]
            image_file_name = img_obj.name
            image_name = os.path.splitext(img_obj.name)[0]
            original_h, original_w = get_hws(cam_obj)
            ixt = get_intrinsics(cam_obj)
            ext = get_extrinsic(img_obj)

        img_path = os.path.join(image_folder, image_file_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        with self.torch.no_grad():
            depth_pred = self.model.infer_image(raw_image, self.processing_res)

        depth_pred = depth_pred.astype(np.float32)
        if depth_pred.shape != (original_h, original_w):
            depth_pred = cv2.resize(depth_pred, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        depth_pred[~np.isfinite(depth_pred)] = 0.0
        depth_pred = np.clip(depth_pred, 0.0, self.max_depth)
        depth_pred, output_ixt, _output_w, _output_h = prepare_depth_output(
            depth_pred,
            ixt,
            original_w,
            original_h,
            self.output_res,
            cv2.INTER_CUBIC,
        )

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
        colorize_depth(depth_pred, self.max_depth).save(color_path)
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


def write_json_line(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_predict(generator, request):
    image_id = int(request["image_id"])
    output_timing = timing_enabled(request)
    gpu_timing = timing_enabled(request.get("gpu_diagnostics", False))
    append_depth_gpu_memory(
        request["output_dir"], "DepthAnythingV2", "before_predict",
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
        request["output_dir"], "DepthAnythingV2", "after_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    if output_timing:
        depth = np.load(result["depth_path"])
        append_depth_worker_timing(
            request["output_dir"], "DepthAnythingV2", image_id,
            request.get("image_name", os.path.splitext(os.path.basename(result["depth_path"]))[0].replace("_depth", "")),
            generator.processing_res, result["width"], result["height"],
            {"total_ms": f"{total_ms:.3f}"},
            generator.torch, generator.device, output_timing)
        append_depth_quality(
            request["output_dir"], "DepthAnythingV2", image_id,
            request.get("image_name", ""), "raw", depth, output_timing)
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
            generator = DepthAnythingV2Generator(
                checkpoint_path=args.checkpoint,
            processing_res=args.processing_res,
            output_res=args.output_res,
                max_depth=args.max_depth,
                device_str=args.device,
                encoder=args.encoder,
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
    generator = DepthAnythingV2Generator(
        checkpoint_path=args.checkpoint,
        processing_res=args.processing_res,
        output_res=args.output_res,
        max_depth=args.max_depth,
        device_str=args.device,
        encoder=args.encoder,
    )
    result = generator.generate_depth(args.image_id, args.sfm_dir, args.image_folder, args.output_dir)
    write_json_line({"ok": True, "image_id": args.image_id, **result})
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DepthAnythingV2 depth prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="DepthAnythingV2 checkpoint directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=784)
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])

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
