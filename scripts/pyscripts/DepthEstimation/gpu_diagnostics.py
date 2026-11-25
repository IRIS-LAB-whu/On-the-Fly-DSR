import csv
import os
import time

import numpy as np


def _dense_reconstruction_dir_from_depth_dir(output_dir):
    return os.path.dirname(os.path.normpath(output_dir))


def _timing_dir(output_dir):
    timing_dir = os.path.join(_dense_reconstruction_dir_from_depth_dir(output_dir), "Timing")
    os.makedirs(timing_dir, exist_ok=True)
    return timing_dir


def _device_is_cuda(device):
    return str(device).lower().startswith("cuda")


def timing_enabled(request_or_enabled=True):
    if isinstance(request_or_enabled, dict):
        value = request_or_enabled.get("output_timing", True)
    else:
        value = request_or_enabled
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _append_csv(output_dir, file_name, row, enabled=True):
    if not timing_enabled(enabled):
        return
    path = os.path.join(_timing_dir(output_dir), file_name)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def torch_memory_row(torch_module=None, device="cpu"):
    row = {
        "cuda_available": False,
        "allocated_mb": "",
        "reserved_mb": "",
        "max_allocated_mb": "",
        "max_reserved_mb": "",
        "mem_free_mb": "",
        "mem_total_mb": "",
    }
    if torch_module is not None and _device_is_cuda(device) and torch_module.cuda.is_available():
        try:
            torch_module.cuda.synchronize()
        except Exception:
            pass
        row["cuda_available"] = True
        row["allocated_mb"] = torch_module.cuda.memory_allocated() / (1024.0 * 1024.0)
        row["reserved_mb"] = torch_module.cuda.memory_reserved() / (1024.0 * 1024.0)
        row["max_allocated_mb"] = torch_module.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        row["max_reserved_mb"] = torch_module.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        if hasattr(torch_module.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            row["mem_free_mb"] = free_bytes / (1024.0 * 1024.0)
            row["mem_total_mb"] = total_bytes / (1024.0 * 1024.0)
    return row


def append_depth_gpu_memory(output_dir, method, event, image_id=None, torch_module=None, device="cpu", enabled=True):
    if not timing_enabled(enabled):
        return
    try:
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "event": event,
            "image_id": "" if image_id is None else image_id,
            "pid": os.getpid(),
            "device": str(device),
        }
        row.update(torch_memory_row(torch_module, device))
        _append_csv(output_dir, "gpu_memory_depth_worker.csv", row, enabled)
    except Exception as exc:
        print(f"[GpuDiagnostics] failed to write depth gpu memory: {exc}", file=os.sys.stderr, flush=True)


def append_depth_worker_timing(
    output_dir,
    method,
    image_id,
    image_name,
    processing_resolution,
    output_width,
    output_height,
    timings,
    torch_module=None,
    device="cpu",
    enabled=True,
):
    try:
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "image_id": image_id,
            "image_name": image_name,
            "processing_resolution": processing_resolution,
            "output_width": output_width,
            "output_height": output_height,
            "preprocess_ms": timings.get("preprocess_ms", ""),
            "inference_ms": timings.get("inference_ms", ""),
            "postprocess_ms": timings.get("postprocess_ms", ""),
            "write_outputs_ms": timings.get("write_outputs_ms", ""),
            "total_ms": timings.get("total_ms", ""),
        }
        row.update(torch_memory_row(torch_module, device))
        _append_csv(output_dir, "depth_worker_timing.csv", row, enabled)
    except Exception as exc:
        print(f"[Timing] failed to write depth worker timing: {exc}", file=os.sys.stderr, flush=True)


def depth_quality_row(method, image_id, image_name, stage, depth):
    depth = np.asarray(depth, dtype=np.float32)
    total = int(depth.size)
    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    zero = finite & (depth == 0.0)
    nan = np.isnan(depth)
    inf = np.isinf(depth)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "image_id": image_id,
        "image_name": image_name,
        "stage": stage,
        "width": int(depth.shape[1]) if depth.ndim == 2 else "",
        "height": int(depth.shape[0]) if depth.ndim == 2 else "",
        "total_pixels": total,
        "valid_ratio": float(np.count_nonzero(positive)) / total if total else 0.0,
        "zero_ratio": float(np.count_nonzero(zero)) / total if total else 0.0,
        "nan_ratio": float(np.count_nonzero(nan)) / total if total else 0.0,
        "inf_ratio": float(np.count_nonzero(inf)) / total if total else 0.0,
        "min": "",
        "max": "",
        "mean": "",
        "median": "",
        "p1": "",
        "p99": "",
    }
    values = depth[positive]
    if values.size > 0:
        row.update({
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p1": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
        })
    return row


def append_depth_quality(output_dir, method, image_id, image_name, stage, depth, enabled=True):
    try:
        _append_csv(
            output_dir,
            "depth_quality_runtime.csv",
            depth_quality_row(method, image_id, image_name, stage, depth),
            enabled,
        )
    except Exception as exc:
        print(f"[Timing] failed to write depth quality: {exc}", file=os.sys.stderr, flush=True)
