import torch


def intrinsics_from_focal_center(fx, fy, cx, cy):
    fx, fy, cx, cy = torch.broadcast_tensors(
        torch.as_tensor(fx),
        torch.as_tensor(fy),
        torch.as_tensor(cx),
        torch.as_tensor(cy),
    )
    intrinsics = torch.zeros((*fx.shape, 3, 3), dtype=fx.dtype, device=fx.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    intrinsics[..., 2, 2] = 1
    return intrinsics


def depth_map_to_point_map(depth, intrinsics):
    if depth.ndim == 2:
        depth_in = depth.unsqueeze(0)
        squeeze_batch = True
    else:
        depth_in = depth
        squeeze_batch = False

    batch, height, width = depth_in.shape
    dtype = depth_in.dtype
    device = depth_in.device
    intrinsics = intrinsics.to(dtype=dtype, device=device)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch, -1, -1)

    u = (torch.arange(width, dtype=dtype, device=device) + 0.5) / width
    v = (torch.arange(height, dtype=dtype, device=device) + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu.unsqueeze(0).expand(batch, -1, -1)
    vv = vv.unsqueeze(0).expand(batch, -1, -1)

    fx = intrinsics[:, 0, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)

    z = depth_in
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    points = torch.stack([x, y, z], dim=-1)
    return points.squeeze(0) if squeeze_batch else points

