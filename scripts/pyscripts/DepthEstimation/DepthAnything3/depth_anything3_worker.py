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
from output_resize_util import prepare_depth_output, resize_map_to_output

DepthAnything3 = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


def load_modules():
    global DepthAnything3
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if DepthAnything3 is not None:
        return

    try:
        from depth_anything_3.api import DepthAnything3 as depth_anything_3_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "DepthAnything3 dependencies are not importable. Expected the copied "
            "depth_anything_3 package and colmap_util.py next to this worker. "
            f"Original import error: {exc}"
        ) from exc

    DepthAnything3 = depth_anything_3_cls
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


def normalize_model_dir(checkpoint_root, model_name):
    root = Path(checkpoint_root)
    model_path = root / model_name
    if model_path.is_dir():
        if (model_path / "model.safetensors").is_file() and (model_path / "config.json").is_file():
            return model_path
        if (model_path / "pytorch_model.bin").is_file() and (model_path / "config.json").is_file():
            return model_path
    if root.is_dir() and (root / "model.safetensors").is_file() and (root / "config.json").is_file():
        return root
    raise FileNotFoundError(
        f"DepthAnything3 checkpoint not found or incomplete: {model_path}. "
        "Expected config.json and model.safetensors."
    )


def resize_optional_map(value, width, height, interpolation):
    if value is None:
        return None
    if value.shape[:2] == (height, width):
        return value
    return cv2.resize(value, (width, height), interpolation=interpolation)


def make_pointmap_from_depth(depth, intrinsic):
    h, w = depth.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depth.astype(np.float32)
    points = np.zeros((h, w, 3), dtype=np.float32)
    if fx > 0 and fy > 0:
        points[..., 0] = (xs - cx) * z / fx
        points[..., 1] = (ys - cy) * z / fy
        points[..., 2] = z
    return points


def has_finite_map(value):
    return value is not None and np.size(value) > 0 and np.isfinite(value).any()


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
        "intrinsic": matrix_from_request(request.get("intrinsic"), 3, 3, "intrinsic").astype(np.float32),
        "pose": matrix_from_request(request.get("pose"), 4, 4, "pose").astype(np.float32),
    }


class DepthAnything3Generator:
    def __init__(
        self,
        checkpoint_path,
        model_name="DA3MONO-LARGE",
        max_depth=50.0,
        process_res=784,
        output_res=784,
        process_res_method="upper_bound_resize",
        save_pointmap=True,
        conf_threshold=0.0,
        device_str="cuda",
    ):
        import torch

        load_modules()
        self.torch = torch
        use_cuda = device_str.lower().startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.max_depth = max_depth
        self.process_res = process_res
        self.output_res = output_res
        self.process_res_method = process_res_method
        self.save_pointmap = save_pointmap
        self.conf_threshold = conf_threshold

        model_dir = normalize_model_dir(checkpoint_path, model_name)
        self.model_name = model_dir.name
        self.model = DepthAnything3.from_pretrained(str(model_dir))
        self.model = self.model.to(device=self.device).eval()

    def generate_depth(self, image_id, sfm_dir, image_folder, output_dir, request=None):
        metadata = metadata_from_request(request)
        if metadata is not None:
            image_file_name = metadata["image_file_name"]
            image_name = metadata["image_name"]
            original_h = metadata["original_h"]
            original_w = metadata["original_w"]
            sfm_ixt = metadata["intrinsic"]
            sfm_ext = metadata["pose"]
        else:
            cams, image, _points = read_model_single_image(sfm_dir, int(image_id))
            img_obj = image
            cam_obj = cams[img_obj.camera_id]
            image_file_name = img_obj.name
            image_name = os.path.splitext(img_obj.name)[0]
            original_h, original_w = get_hws(cam_obj)
            sfm_ixt = get_intrinsics(cam_obj).astype(np.float32)
            sfm_ext = get_extrinsic(img_obj).astype(np.float32)

        img_path = os.path.join(image_folder, image_file_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        inference_kwargs = {
            "image": [img_path],
            "process_res": int(self.process_res),
            "process_res_method": self.process_res_method,
        }

        with self.torch.no_grad():
            prediction = self.model.inference(**inference_kwargs)

        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        conf = None
        if getattr(prediction, "conf", None) is not None:
            conf = np.asarray(prediction.conf[0], dtype=np.float32)
        da3_ixt = None
        if getattr(prediction, "intrinsics", None) is not None:
            da3_ixt = np.asarray(prediction.intrinsics[0], dtype=np.float64)
        da3_ext = None
        if getattr(prediction, "extrinsics", None) is not None:
            da3_ext = np.asarray(prediction.extrinsics[0], dtype=np.float64)

        depth[~np.isfinite(depth)] = 0.0
        depth = resize_optional_map(depth, original_w, original_h, cv2.INTER_LINEAR).astype(np.float32)
        if conf is not None:
            conf = resize_optional_map(conf, original_w, original_h, cv2.INTER_LINEAR).astype(np.float32)

        valid_depth = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth)
        if conf is not None and self.conf_threshold > 0:
            valid_depth &= np.isfinite(conf) & (conf >= self.conf_threshold)
        depth[~valid_depth] = 0.0
        depth = np.clip(depth, 0.0, self.max_depth).astype(np.float32)
        depth, output_ixt, output_w, output_h = prepare_depth_output(
            depth,
            sfm_ixt,
            original_w,
            original_h,
            self.output_res,
            cv2.INTER_LINEAR,
        )
        if conf is not None:
            conf = resize_map_to_output(conf, output_w, output_h, cv2.INTER_LINEAR).astype(np.float32)
        valid_depth = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth)
        if conf is not None and self.conf_threshold > 0:
            valid_depth &= np.isfinite(conf) & (conf >= self.conf_threshold)
        depth[~valid_depth] = 0.0

        # Keep the display pointmap in the SfM camera frame used by the rest of the system.
        pointmap = make_pointmap_from_depth(depth, output_ixt.astype(np.float32))
        pointmap[~valid_depth] = 0.0

        np_output = os.path.join(output_dir, "np")
        color_output = os.path.join(output_dir, "color")
        intrinsic_output = os.path.join(output_dir, "intrinsic")
        pose_output = os.path.join(output_dir, "pose")
        pointmap_output = os.path.join(output_dir, "pointmap")
        mask_output = os.path.join(output_dir, "mask")
        conf_output = os.path.join(output_dir, "conf")
        da3_intrinsic_output = os.path.join(output_dir, "da3_intrinsic")
        da3_pose_output = os.path.join(output_dir, "da3_pose")

        for path in (np_output, color_output, intrinsic_output, pose_output, mask_output):
            os.makedirs(path, exist_ok=True)
        if self.save_pointmap:
            for path in (pointmap_output, conf_output, da3_intrinsic_output, da3_pose_output):
                os.makedirs(path, exist_ok=True)

        depth_path = os.path.join(np_output, f"{image_name}_depth.npy")
        color_path = os.path.join(color_output, f"{image_name}_color.png")
        intrinsic_path = os.path.join(intrinsic_output, f"{image_name}.txt")
        pose_path = os.path.join(pose_output, f"{image_name}.txt")
        mask_path = os.path.join(mask_output, f"{image_name}_mask.png")

        np.save(depth_path, depth)
        colorize_depth(depth, self.max_depth).save(color_path)
        np.savetxt(intrinsic_path, output_ixt.astype(np.float64))
        np.savetxt(pose_path, sfm_ext.astype(np.float64))
        cv2.imwrite(mask_path, (valid_depth.astype(np.uint8) * 255))

        pointmap_path = ""
        conf_path = ""
        da3_intrinsic_path = ""
        da3_pose_path = ""
        if self.save_pointmap:
            pointmap_path = os.path.join(pointmap_output, f"{image_name}_points.npy")
            np.save(pointmap_path, pointmap.astype(np.float32))
            if has_finite_map(conf):
                conf_path = os.path.join(conf_output, f"{image_name}_conf.npy")
                np.save(conf_path, conf.astype(np.float32))
            if da3_ixt is not None:
                da3_intrinsic_path = os.path.join(da3_intrinsic_output, f"{image_name}.txt")
                np.savetxt(da3_intrinsic_path, da3_ixt)
            if da3_ext is not None:
                da3_pose_path = os.path.join(da3_pose_output, f"{image_name}.txt")
                np.savetxt(da3_pose_path, da3_ext)

        return {
            "depth_path": depth_path,
            "color_path": color_path,
            "intrinsic_path": intrinsic_path,
            "pose_path": pose_path,
            "mask_path": mask_path,
            "pointmap_path": pointmap_path,
            "conf_path": conf_path,
            "da3_intrinsic_path": da3_intrinsic_path,
            "da3_pose_path": da3_pose_path,
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
        request["output_dir"], "DepthAnything3", "before_predict",
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
        request["output_dir"], "DepthAnything3", "after_predict",
        image_id, generator.torch, generator.device, gpu_timing)
    if output_timing:
        image_name = request.get("image_name", "")
        depth = np.load(result["depth_path"])
        append_depth_worker_timing(
            request["output_dir"], "DepthAnything3", image_id, image_name,
            generator.process_res, result["width"], result["height"],
            {"total_ms": f"{total_ms:.3f}"},
            generator.torch, generator.device, output_timing)
        append_depth_quality(
            request["output_dir"], "DepthAnything3", image_id, image_name,
            "raw", depth, output_timing)
    return {
        "type": "result",
        "request_id": request.get("request_id"),
        "image_id": image_id,
        "ok": True,
        **result,
    }


def create_generator(args):
    return DepthAnything3Generator(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        max_depth=args.max_depth,
        process_res=args.processing_res,
        output_res=args.output_res,
        process_res_method=args.process_res_method,
        save_pointmap=args.save_pointmap,
        conf_threshold=args.conf_threshold,
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
                write_json_line({
                    "type": "shutdown",
                    "request_id": request.get("request_id"),
                    "ok": True,
                })
                return 0
            if request_type != "predict":
                raise ValueError(f"Unsupported request type: {request_type}")
            response = handle_predict(generator, request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "type": "result",
                "request_id": request.get("request_id") if "request" in locals() else None,
                "ok": False,
                "error": str(exc),
            }
        write_json_line(response)
    return 0


def run_single(args):
    generator = create_generator(args)
    result = generator.generate_depth(
        image_id=args.image_id,
        sfm_dir=args.sfm_dir,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DepthAnything3 prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="DepthAnything3 checkpoint root directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=784)
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--denoise-steps", type=int, default=0, help="Reserved for interface compatibility.")
    parser.add_argument("--ensemble-size", type=int, default=0, help="Reserved for interface compatibility.")
    parser.add_argument("--model", default="DA3MONO-LARGE")
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--save-pointmap", type=parse_bool, default=True)
    parser.add_argument("--conf-threshold", type=float, default=0.0)

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
