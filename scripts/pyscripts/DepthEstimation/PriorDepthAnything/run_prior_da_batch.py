import argparse
import json
import os
import sys

from colmap_util import read_model
from prior_depth_anything_worker import PriorDepthAnythingGenerator


def write_progress(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch PriorDepthAnything depth prediction.")
    parser.add_argument("--checkpoint", required=True, help="PriorDepthAnything checkpoint directory.")
    parser.add_argument("--sfm-dir", required=True, help="COLMAP sparse model directory.")
    parser.add_argument("--image-folder", required=True, help="Input image folder.")
    parser.add_argument("--output-dir", required=True, help="Depth output directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--processing-res", type=int, default=0)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--version", default="1.1", choices=["1.0", "1.1"])
    parser.add_argument("--frozen-model-size", default="vitb", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--conditioned-model-size", default="vitb", choices=["vits", "vitb"])
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.sfm_dir):
        raise FileNotFoundError(f"SfM directory not found: {args.sfm_dir}")
    if not os.path.isdir(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")

    _cams, images, _points = read_model(args.sfm_dir)
    image_ids = sorted(images.keys())
    if args.start_id is not None:
        image_ids = [image_id for image_id in image_ids if image_id >= args.start_id]
    if args.end_id is not None:
        image_ids = [image_id for image_id in image_ids if image_id <= args.end_id]

    generator = PriorDepthAnythingGenerator(
        checkpoint_path=args.checkpoint,
        processing_res=args.processing_res,
        max_depth=args.max_depth,
        device_str=args.device,
        version=args.version,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        pattern=args.pattern,
    )

    write_progress({
        "type": "ready",
        "count": len(image_ids),
        "sfm_dir": args.sfm_dir,
        "image_folder": args.image_folder,
        "output_dir": args.output_dir,
    })

    ok_count = 0
    failed_count = 0
    for index, image_id in enumerate(image_ids, start=1):
        image = images[image_id]
        image_stem = os.path.splitext(image.name)[0]
        depth_path = os.path.join(args.output_dir, "np", f"{image_stem}_depth.npy")
        if args.skip_existing and os.path.isfile(depth_path):
            write_progress({
                "type": "skip",
                "index": index,
                "count": len(image_ids),
                "image_id": image_id,
                "image": image.name,
                "depth": depth_path,
            })
            ok_count += 1
            continue

        try:
            result = generator.generate_depth(
                image_id=image_id,
                sfm_dir=args.sfm_dir,
                image_folder=args.image_folder,
                output_dir=args.output_dir,
            )
            ok_count += 1
            write_progress({
                "type": "result",
                "index": index,
                "count": len(image_ids),
                "image_id": image_id,
                "image": image.name,
                **result,
            })
        except Exception as exc:
            failed_count += 1
            write_progress({
                "type": "error",
                "index": index,
                "count": len(image_ids),
                "image_id": image_id,
                "image": image.name,
                "message": str(exc),
            })

    write_progress({"type": "done", "ok": ok_count, "failed": failed_count})
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
