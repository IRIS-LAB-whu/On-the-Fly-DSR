import cv2
import numpy as np


def compute_output_size(original_w, original_h, output_res):
    original_w = int(original_w)
    original_h = int(original_h)
    output_res = int(output_res or 0)
    if output_res <= 0 or max(original_w, original_h) <= 0:
        return original_w, original_h
    scale = float(output_res) / float(max(original_w, original_h))
    out_w = max(1, int(round(original_w * scale)))
    out_h = max(1, int(round(original_h * scale)))
    return out_w, out_h


def scale_intrinsic_for_output(intrinsic, original_w, original_h, output_w, output_h):
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    sx = float(output_w) / float(original_w)
    sy = float(output_h) / float(original_h)
    scaled[0, 0] *= sx
    scaled[0, 1] *= sx
    scaled[0, 2] *= sx
    scaled[1, 0] *= sy
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def resize_map_to_output(value, output_w, output_h, interpolation):
    if value is None:
        return None
    if value.shape[:2] == (int(output_h), int(output_w)):
        return value
    return cv2.resize(value, (int(output_w), int(output_h)), interpolation=interpolation)


def prepare_depth_output(depth, intrinsic, original_w, original_h, output_res, interpolation=cv2.INTER_LINEAR):
    output_w, output_h = compute_output_size(original_w, original_h, output_res)
    depth_out = resize_map_to_output(depth, output_w, output_h, interpolation).astype(np.float32)
    intrinsic_out = scale_intrinsic_for_output(intrinsic, original_w, original_h, output_w, output_h)
    return depth_out, intrinsic_out, output_w, output_h


def aligned_size_for_patch(width, height, patch_divisor):
    patch_divisor = int(patch_divisor)
    if patch_divisor <= 1:
        return int(width), int(height)
    aligned_w = int(width) - int(width) % patch_divisor
    aligned_h = int(height) - int(height) % patch_divisor
    return max(1, aligned_w), max(1, aligned_h)


def prepare_cropped_depth_output(
    depth,
    intrinsic,
    original_w,
    original_h,
    output_res,
    patch_divisor,
    interpolation=cv2.INTER_LINEAR,
):
    output_w, output_h = compute_output_size(original_w, original_h, output_res)
    valid_w, valid_h = aligned_size_for_patch(output_w, output_h, patch_divisor)
    depth_valid = resize_map_to_output(depth, valid_w, valid_h, interpolation).astype(np.float32)
    depth_out = np.zeros((output_h, output_w), dtype=np.float32)
    depth_out[:valid_h, :valid_w] = depth_valid
    intrinsic_out = scale_intrinsic_for_output(intrinsic, original_w, original_h, output_w, output_h)
    return depth_out, intrinsic_out, output_w, output_h, valid_w, valid_h
