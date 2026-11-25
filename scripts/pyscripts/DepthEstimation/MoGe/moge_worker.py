import argparse
import contextlib
import json
import math
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

MoGeModel = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


def load_modules():
    global MoGeModel
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if MoGeModel is not None:
        return

    try:
        from moge.model.v2 import MoGeModel as moge_model_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "MoGe dependencies are not importable. Expected the moge package "
            "and colmap_util.py next to this worker. "
            f"Original import error: {exc}"
        ) from exc

    MoGeModel = moge_model_cls
    get_extrinsic = get_extrinsic_func
    get_hws = get_hws_func
    get_intrinsics = get_intrinsics_func
    read_model = read_model_func
    read_model_single_image = read_model_single_image_func


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
        gray[~valid] = 0
        return Image.fromarray(gray).convert("RGB")


def normalize_model_path(checkpoint_root, model_name):
    model_path = Path(checkpoint_root) / model_name
    if model_path.is_dir():
        model_file = model_path / "model.pt"
        if model_file.is_file():
            return str(model_file)
    if model_path.is_file():
        return str(model_path)
    raise FileNotFoundError(f"MoGe checkpoint not found: {model_path}")


def fov_x_from_intrinsics(intrinsics, width):
    fx = float(intrinsics[0, 0])
    if fx <= 0:
        return None
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * fx)))


def resize_optional_map(value, width, height, interpolation):
    if value is None:
        return None
    if value.shape[:2] == (height, width):
        return value
    return cv2.resize(value, (width, height), interpolation=interpolation)


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


class MoGeDepthGenerator:
    def __init__(
        self,
        checkpoint_path,
        model_name="moge-2-vitl-normal",
        max_depth=50.0,
        output_res=784,
        resolution_level=9,
        num_tokens=0,
        use_fp16=True,
        use_sfm_fov=True,
        apply_mask=True,
        mask_threshold=0.5,
        save_pointmap=True,
        save_normal=True,
        device_str="cuda",
    ):
        import torch

        load_modules()
        self.torch = torch
        use_cuda = device_str.lower().startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.max_depth = max_depth
        self.output_res = output_res
        self.resolution_level = resolution_level
        self.num_tokens = None if num_tokens <= 0 else int(num_tokens)
        self.use_fp16 = use_fp16
        self.use_sfm_fov = use_sfm_fov
        self.apply_mask = apply_mask
        self.mask_threshold = mask_threshold
        self.save_pointmap = save_pointmap
        self.save_normal = save_normal

        pretrained = normalize_model_path(checkpoint_path, model_name)
        self.model = MoGeModel.from_pretrained(pretrained).to(self.device).eval()
        if self.use_fp16 and self.device.type == "cuda":
            self.model.half()

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
        fov_x = fov_x_from_intrinsics(ixt, original_w) if self.use_sfm_fov else None

        raw_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.torch.tensor(
            raw_rgb / 255.0,
            dtype=self.torch.float32,
            device=self.device).permute(2, 0, 1)

        with self.torch.no_grad():
            output = self.model.infer(
                image_tensor,
                fov_x=fov_x,
                resolution_level=self.resolution_level,
                num_tokens=self.num_tokens,
                apply_mask=self.apply_mask,
                mask_threshold=self.mask_threshold,
                use_fp16=self.use_fp16 and self.device.type == "cuda")

        depth = output["depth"].detach().cpu().numpy().astype(np.float32)
        mask = output.get("mask")
        mask_np = mask.detach().cpu().numpy().astype(bool) if mask is not None else np.isfinite(depth)
        points = output.get("points")
        points_np = points.detach().cpu().numpy().astype(np.float32) if points is not None else None
        normal = output.get("normal")
        normal_np = normal.detach().cpu().numpy().astype(np.float32) if normal is not None else None
        moge_intrinsics = output.get("intrinsics")
        moge_ixt_np = moge_intrinsics.detach().cpu().numpy().astype(np.float64) if moge_intrinsics is not None else None

        depth[~np.isfinite(depth)] = 0.0
        if points_np is not None:
            points_np[~np.isfinite(points_np)] = 0.0
        if normal_np is not None:
            normal_np[~np.isfinite(normal_np)] = 0.0

        depth = resize_optional_map(depth, original_w, original_h, cv2.INTER_LINEAR).astype(np.float32)
        mask_np = resize_optional_map(mask_np.astype(np.uint8), original_w, original_h, cv2.INTER_NEAREST).astype(bool)
        points_np = resize_optional_map(points_np, original_w, original_h, cv2.INTER_LINEAR)
        normal_np = resize_optional_map(normal_np, original_w, original_h, cv2.INTER_LINEAR)

        filter_mask = mask_np if self.apply_mask else np.ones_like(mask_np, dtype=bool)
        valid_depth = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth) & filter_mask
        depth[~valid_depth] = 0.0
        depth = np.clip(depth, 0.0, self.max_depth).astype(np.float32)
        if points_np is not None:
            points_np[~valid_depth] = 0.0
        if normal_np is not None:
            normal_np[~valid_depth] = 0.0
        depth, output_ixt, output_w, output_h = prepare_depth_output(
            depth,
            ixt,
            original_w,
            original_h,
            self.output_res,
            cv2.INTER_LINEAR,
        )
        mask_np = resize_optional_map(mask_np.astype(np.uint8), output_w, output_h, cv2.INTER_NEAREST).astype(bool)
        valid_depth = resize_optional_map(valid_depth.astype(np.uint8), output_w, output_h, cv2.INTER_NEAREST).astype(bool)
        points_np = resize_optional_map(points_np, output_w, output_h, cv2.INTER_LINEAR)
        normal_np = resize_optional_map(normal_np, output_w, output_h, cv2.INTER_LINEAR)
        depth[~valid_depth] = 0.0
        if points_np is not None:
            points_np[~valid_depth] = 0.0
        if normal_np is not None:
            normal_np[~valid_depth] = 0.0

        np_output = os.path.join(output_dir, "np")
        color_output = os.path.join(output_dir, "color")
        intrinsic_output = os.path.join(output_dir, "intrinsic")
        pose_output = os.path.join(output_dir, "pose")
        pointmap_output = os.path.join(output_dir, "pointmap")
        mask_output = os.path.join(output_dir, "mask")
        normal_output = os.path.join(output_dir, "normal")
        moge_intrinsic_output = os.path.join(output_dir, "moge_intrinsic")

        for path in (np_output, color_output, intrinsic_output, pose_output, mask_output, moge_intrinsic_output):
            os.makedirs(path, exist_ok=True)
        if self.save_pointmap:
            os.makedirs(pointmap_output, exist_ok=True)
        if self.save_normal and normal_np is not None:
            os.makedirs(normal_output, exist_ok=True)

        depth_path = os.path.join(np_output, f"{image_name}_depth.npy")
        color_path = os.path.join(color_output, f"{image_name}_color.png")
        intrinsic_path = os.path.join(intrinsic_output, f"{image_name}.txt")
        pose_path = os.path.join(pose_output, f"{image_name}.txt")
        mask_path = os.path.join(mask_output, f"{image_name}_mask.png")
        moge_intrinsic_path = os.path.join(moge_intrinsic_output, f"{image_name}.txt")

        np.save(depth_path, depth)
        colorize_depth(depth, self.max_depth).save(color_path)
        np.savetxt(intrinsic_path, output_ixt)
        np.savetxt(pose_path, ext)
        cv2.imwrite(mask_path, (valid_depth.astype(np.uint8) * 255))
        if moge_ixt_np is not None:
            np.savetxt(moge_intrinsic_path, moge_ixt_np)

        pointmap_path = ""
        if self.save_pointmap and points_np is not None:
            pointmap_path = os.path.join(pointmap_output, f"{image_name}_points.npy")
            np.save(pointmap_path, points_np.astype(np.float32))

        normal_path = ""
        if self.save_normal and normal_np is not None:
            normal_path = os.path.join(normal_output, f"{image_name}_normal.npy")
            np.save(normal_path, normal_np.astype(np.float32))

        return {
            "depth_path": depth_path,
            "color_path": color_path,
            "intrinsic_path": intrinsic_path,
            "pose_path": pose_path,
            "mask_path": mask_path,
            "pointmap_path": pointmap_path,
            "normal_path": normal_path,
            "moge_intrinsic_path": moge_intrinsic_path if moge_ixt_np is not None else "",
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
        }

    def generate_depth_from_name(self, image_name, sfm_dir, image_folder, output_dir):
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
        request["output_dir"], "MoGe", "before_predict",
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
        request["output_dir"], "MoGe", "after_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    if output_timing:
        image_name = request.get("image_name", "")
        depth = np.load(result["depth_path"])
        append_depth_worker_timing(
            request["output_dir"], "MoGe", image_id, image_name,
            generator.resolution_level, result["width"], result["height"],
            {"total_ms": f"{total_ms:.3f}"},
            generator.torch, generator.device, output_timing)
        append_depth_quality(
            request["output_dir"], "MoGe", image_id, image_name,
            "raw", depth, output_timing)
    return {
        "type": "result",
        "request_id": request.get("request_id"),
        "image_id": image_id,
        "ok": True,
        **result,
    }


def create_generator(args):
    return MoGeDepthGenerator(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        max_depth=args.max_depth,
        output_res=args.output_res,
        resolution_level=args.resolution_level,
        num_tokens=args.num_tokens,
        use_fp16=args.fp16,
        use_sfm_fov=args.use_sfm_fov,
        apply_mask=args.apply_mask,
        mask_threshold=args.mask_threshold,
        save_pointmap=args.save_pointmap,
        save_normal=args.save_normal,
        device_str=args.device,
    )


def run_worker(args):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            generator = create_generator(args)
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
    generator = create_generator(args)
    result = generator.generate_depth(args.image_id, args.sfm_dir, args.image_folder, args.output_dir)
    write_json_line({"ok": True, "image_id": args.image_id, **result})
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MoGe2 depth and point-map prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="MoGe checkpoint root directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=9, help="Alias for --resolution-level.")
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--denoise-steps", type=int, default=0, help="Reserved for interface compatibility.")
    parser.add_argument("--ensemble-size", type=int, default=0, help="Reserved for interface compatibility.")
    parser.add_argument("--model", default="moge-2-vitl-normal")
    parser.add_argument("--resolution-level", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=0)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--use-sfm-fov", type=parse_bool, default=True)
    parser.add_argument("--apply-mask", type=parse_bool, default=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--save-pointmap", type=parse_bool, default=True)
    parser.add_argument("--save-normal", type=parse_bool, default=True)

    parser.add_argument("--image-id", type=int, help="Single-shot COLMAP image id.")
    parser.add_argument("--sfm-dir", help="Single-shot COLMAP sparse model directory.")
    parser.add_argument("--image-folder", help="Single-shot image folder.")
    parser.add_argument("--output-dir", help="Single-shot output directory.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.resolution_level is None:
        args.resolution_level = args.processing_res
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
