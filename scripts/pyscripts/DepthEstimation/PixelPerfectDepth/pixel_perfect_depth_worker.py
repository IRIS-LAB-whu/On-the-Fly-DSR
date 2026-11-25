import argparse
import contextlib
import json
import os
import random
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
from output_resize_util import compute_output_size, prepare_depth_output

PixelPerfectDepth = None
get_extrinsic = None
get_hws = None
get_intrinsics = None
read_model = None
read_model_single_image = None


def load_modules():
    global PixelPerfectDepth
    global get_extrinsic
    global get_hws
    global get_intrinsics
    global read_model
    global read_model_single_image

    if PixelPerfectDepth is not None:
        return

    try:
        from ppd.models.ppd import PixelPerfectDepth as ppd_cls
        from colmap_util import get_extrinsic as get_extrinsic_func
        from colmap_util import get_hws as get_hws_func
        from colmap_util import get_intrinsics as get_intrinsics_func
        from colmap_util import read_model as read_model_func
        from colmap_util import read_model_single_image as read_model_single_image_func
    except ImportError as exc:
        raise ImportError(
            "PixelPerfectDepth dependencies are not importable. Expected the "
            "ppd package and colmap_util.py next to this worker. "
            f"Original import error: {exc}"
        ) from exc

    PixelPerfectDepth = ppd_cls
    get_extrinsic = get_extrinsic_func
    get_hws = get_hws_func
    get_intrinsics = get_intrinsics_func
    read_model = read_model_func
    read_model_single_image = read_model_single_image_func


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def normalize_semantics_model(value):
    lowered = str(value).strip().lower()
    if lowered in ("moge2", "moge-2", "moge"):
        return "MoGe2"
    if lowered in ("da2", "depthanythingv2", "depth-anything-v2", "depth_anything_v2"):
        return "DA2"
    raise ValueError(f"Unsupported PixelPerfectDepth semantics model: {value}")


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


class PixelPerfectDepthGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        processing_res: int = 784,
        output_res: int = 784,
        max_depth: float = 50.0,
        sampling_steps: int = 4,
        device_str: str = "cuda",
        use_fp16: bool = True,
        semantics_model: str = "DA2",
        seed: int = 666,
    ):
        import torch

        load_modules()
        set_seed(seed)

        self.torch = torch
        self.processing_res = processing_res
        self.output_res = output_res
        self.max_depth = max_depth
        self.use_fp16 = use_fp16
        self.semantics_model = normalize_semantics_model(semantics_model)

        checkpoint_dir = Path(checkpoint_path)
        if self.semantics_model == "MoGe2":
            ppd_checkpoint = checkpoint_dir / "ppd_moge.pth"
            semantics_checkpoint = checkpoint_dir / "moge2.pt"
        else:
            ppd_checkpoint = checkpoint_dir / "ppd.pth"
            semantics_checkpoint = checkpoint_dir / "depth_anything_v2_vitl.pth"
        if not ppd_checkpoint.is_file():
            raise FileNotFoundError(f"PixelPerfectDepth checkpoint not found: {ppd_checkpoint}")
        if not semantics_checkpoint.is_file():
            raise FileNotFoundError(
                f"PixelPerfectDepth semantic checkpoint not found: {semantics_checkpoint}"
            )

        self.model = PixelPerfectDepth(
            semantics_model=self.semantics_model,
            semantics_pth=str(semantics_checkpoint),
            sampling_steps=sampling_steps,
            device_str=device_str,
        )
        self.model.load_state_dict(torch.load(str(ppd_checkpoint), map_location="cpu"), strict=False)
        self.model = self.model.to(self.model.device).eval()

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
        if self.processing_res > 0:
            proc_w, proc_h = compute_output_size(original_w, original_h, self.processing_res)
            if raw_image.shape[:2] != (proc_h, proc_w):
                raw_image = cv2.resize(raw_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        with self.torch.no_grad():
            depth_pred, _resized_image = self.model.infer_image(raw_image, use_fp16=self.use_fp16)

        depth_pred = depth_pred.squeeze().detach().cpu().numpy().astype(np.float32)
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
        request["output_dir"], "PixelPerfectDepth", "before_predict",
        image_id, generator.torch, generator.model.device, gpu_timing)
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
        request["output_dir"], "PixelPerfectDepth", "after_predict",
        image_id, generator.torch, generator.model.device, gpu_timing)
    if output_timing:
        image_name = request.get("image_name", "")
        depth = np.load(result["depth_path"])
        append_depth_worker_timing(
            request["output_dir"], "PixelPerfectDepth", image_id, image_name,
            generator.processing_res, result["width"], result["height"],
            {"total_ms": f"{total_ms:.3f}"},
            generator.torch, generator.model.device, output_timing)
        append_depth_quality(
            request["output_dir"], "PixelPerfectDepth", image_id, image_name,
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
            generator = PixelPerfectDepthGenerator(
                checkpoint_path=args.checkpoint,
                processing_res=args.processing_res,
                output_res=args.output_res,
                max_depth=args.max_depth,
                sampling_steps=args.denoise_steps,
                device_str=args.device,
                use_fp16=args.fp16,
                semantics_model=args.semantics_model,
                seed=args.seed,
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
    generator = PixelPerfectDepthGenerator(
        checkpoint_path=args.checkpoint,
        processing_res=args.processing_res,
        output_res=args.output_res,
        max_depth=args.max_depth,
        sampling_steps=args.denoise_steps,
        device_str=args.device,
        use_fp16=args.fp16,
        semantics_model=args.semantics_model,
        seed=args.seed,
    )
    result = generator.generate_depth(args.image_id, args.sfm_dir, args.image_folder, args.output_dir)
    write_json_line({"ok": True, "image_id": args.image_id, **result})
    return 0


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="PixelPerfectDepth prediction worker.")
    parser.add_argument("--worker", action="store_true", help="Run as a persistent JSON-lines worker.")
    parser.add_argument("--checkpoint", required=True, help="PixelPerfectDepth checkpoint directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--processing-res", type=int, default=784)
    parser.add_argument("--output-res", type=int, default=784)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--denoise-steps", type=int, default=4, help="PixelPerfectDepth sampling steps.")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Reserved for interface compatibility.")
    parser.add_argument("--fp16", type=parse_bool, default=True, help="Use CUDA mixed precision.")
    parser.add_argument("--semantics-model", default="DA2")
    parser.add_argument("--seed", type=int, default=666)

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
