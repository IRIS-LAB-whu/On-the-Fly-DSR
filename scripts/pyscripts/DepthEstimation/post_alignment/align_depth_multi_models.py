# -*- coding: utf-8 -*-
"""
项目名称：基于摄影测量约束的单目深度估计 (MDE) 尺度校正系统
版本：本科毕业论文-后校正-最终版
主要功能：
    1. 尺度恢复：利用 SfM 稀疏点云作为绝对参考，通过 RANSAC 拟合实现单目深度估计的物理尺度还原。
    2. 分块校正：通过滑动窗口局部拟合，解决单目模型在大场景下的局部几何形变问题。
    3. 缝隙消除：采用 2D 余弦窗口权重融合技术，确保分块边缘在 3D 重建中无接缝。
    4. 鲁棒性优化：内置 Z-buffer 遮挡处理、动态 RANSAC 采样及权重Fallback机制。
"""

import os
import argparse
import numpy as np
import cv2
import csv
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from concurrent.futures import ProcessPoolExecutor

# 导入 COLMAP 专用读取模块
try:
    from colmap_read_model import read_model, qvec2rotmat
except ImportError:
    print("❌ Error: 未能在路径中找到 colmap_read_model.py，请检查脚本完整性。")
    exit(1)

# ==============================================================================
# 【一、全局计算配置】
# ==============================================================================
MIN_POINTS_GLOBAL = 15     # 全局对齐所需的最小有效 SfM 观测数
MIN_POINTS_BLOCK  = 10     # 每个分块局部拟合所需的最小观测数
DEPTH_MIN = 1e-3           # 数值计算的深度下限，防止除零
EPS = 1e-6                 # 极小偏移量，用于对数空间稳定性
MAX_DEPTH = 200.0          # 深度截断上限，防止远景无限拉伸

def safe_log(x):
    """数值稳定的对数变换"""
    return np.log(np.clip(x, DEPTH_MIN, None))

def safe_inv(x):
    """数值稳定的逆深度（视差空间）变换"""
    return 1.0 / np.clip(x, DEPTH_MIN, None)

# ==============================================================================
# 【二、鲁棒数学拟合算子】
# ==============================================================================
def fit_model(x, y, mode):
    """
    核心拟合引擎：在不同数学流形空间进行尺度映射
    Args:
        x: MDE 原始预测值, y: SfM 参考值, mode: 几何模型类型
    Returns:
        model: RANSAC 对象, mask: 内点掩码
    """
    try:
        if mode == "linear":
            X, Y, th = x, y, 0.5
            model = LinearRegression()
        elif mode == "inverse":
            X, Y, th = x, safe_inv(y), 0.1
            model = LinearRegression()
        elif mode == "log":
            X, Y, th = safe_log(x), safe_log(y), 0.2
            model = LinearRegression()
        elif mode == "quadratic":
            X, Y, th = x, y, 0.5
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        elif mode == "inv_quad":
            X, Y, th = x, safe_inv(y), 0.1
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        else:
            return None, None

        # 优化：动态 RANSAC 采样策略 (兼顾小样本准确度与大样本稳定性)
        n_samples = max(6, int(0.2 * len(x)))
        
        ransac = RANSACRegressor(
            model,
            residual_threshold=th,
            min_samples=min(n_samples, len(x)),
            random_state=42
        )
        ransac.fit(X, Y)
        return ransac, ransac.inlier_mask_
    except Exception:
        return None, None

def predict(model, x, mode):
    """逆变换预测器：将拟合结果还原回物理深度（米）"""
    X = safe_log(x) if mode == "log" else x
    y = model.predict(X)
    if mode in ["inverse", "inv_quad"]:
        return safe_inv(y)
    elif mode == "log":
        return np.exp(y)
    return y

def select_best(x, y, modes):
    """基于物理空间 RMSE 进行模型自动选拔"""
    best_rmse = np.inf
    best = None
    stats = {}

    for m in modes:
        model, mask = fit_model(x, y, m)
        if model is None or mask is None or np.sum(mask) < 5:
            continue
        p = predict(model, x, m)
        # 仅针对 RANSAC 内点计算 RMSE，反映真实拟合质量
        rmse = np.sqrt(np.mean((y[mask] - p[mask])**2))
        
        stats[m] = {
            "rmse": rmse,
            "inlier_ratio": np.mean(mask),
            "num_inliers": int(np.sum(mask))
        }
        
        if rmse < best_rmse:
            best_rmse = rmse
            best = (model, m)
    return best, stats

# ==============================================================================
# 【三、滑动窗口精细化对齐逻辑】
# ==============================================================================
def apply_blocks(d_mono, sparse, mask, model_g, mode_g, grid, overlap, local_fit, local_model_mode):
    """
    分块自适应校正算法
    Args:
        local_fit: 是否执行局部拟合
        local_model_mode: "global" (锁定模型名) 或 "per_block" (独立选拔模型)
    """
    h, w = d_mono.shape
    if grid == 1:
        return predict(model_g, d_mono.reshape(-1,1), mode_g).reshape(h,w)

    out = np.zeros((h,w))
    weight = np.zeros((h,w))

    # 计算步长，确保高重叠率下 sh/sw 不为 0 导致死循环
    bh, bw = h // grid, w // grid
    sh = max(1, int(bh * (1 - overlap)))
    sw = max(1, int(bw * (1 - overlap)))

    # 生成 2D 余弦权重窗，用于消除分块接缝
    win = np.outer(np.sin(np.linspace(0, np.pi, bh)), np.sin(np.linspace(0, np.pi, bw)))
    modes = ["linear","inverse","log","quadratic","inv_quad"]

    for y in range(0, h - bh + 1, sh):
        for x in range(0, w - bw + 1, sw):
            y1, x1 = y + bh, x + bw
            bm, bv = d_mono[y:y1, x:x1], mask[y:y1, x:x1]

            # 只有当开启 local_fit 且块内点数足够时才进行局部拟合
            if local_fit and np.sum(bv) > MIN_POINTS_BLOCK * 2:
                # 性能优化：限制块内计算点数，防止极高分辨率下变慢
                if np.sum(bv) > 300:
                    idx = np.where(bv.flatten())[0]
                    sample_idx = np.random.choice(idx, 300, replace=False)
                    bx_in = d_mono[y:y1, x:x1].flatten()[sample_idx].reshape(-1,1)
                    by_tg = sparse[y:y1, x:x1].flatten()[sample_idx]
                else:
                    bx_in = bm[bv].reshape(-1,1)
                    by_tg = sparse[y:y1, x:x1][bv]

                if local_model_mode == "global":
                    m, _ = fit_model(bx_in, by_tg, mode_g)
                    curr_m, curr_mode = (m, mode_g) if m else (model_g, mode_g)
                else:
                    best, _ = select_best(bx_in, by_tg, modes)
                    curr_m, curr_mode = best if best else (model_g, mode_g)
            else:
                curr_m, curr_mode = model_g, mode_g

            p = predict(curr_m, bm.reshape(-1,1), curr_mode).reshape(bh, bw)
            out[y:y1, x:x1] += p * win
            weight[y:y1, x:x1] += win

    # 权重归一化与 Fallback 填充
    valid_w = weight > EPS
    out[valid_w] /= weight[valid_w]
    
    if not np.all(valid_w):
        # 对未被窗口完全覆盖的边缘部分采用全局对齐结果填充
        g_full = predict(model_g, d_mono.reshape(-1,1), mode_g).reshape(h,w)
        out[~valid_w] = g_full[~valid_w]

    return out

# ==============================================================================
# 【四、单帧影像流水线函数】
# ==============================================================================
def process(task):
    """
    单帧影像执行流：数据加载 -> Z-buffer 重投影 -> 统计优选 -> 分块校正
    """
    (name, qvec, tvec, h, w, pts_raw, xys_raw, npy_dir, out_dir, mode, grid, overlap, local_fit, l_mode) = task
    base = os.path.splitext(name)[0]
    candidates = [os.path.join(npy_dir, base+suffix) for suffix in [".npy","_depth.npy","_pred.npy"]]
    npy_path = next((p for p in candidates if os.path.exists(p)), None)
    if not npy_path: return False, "NPY缺失", None, None

    # 1. 深度图加载
    d = np.load(npy_path).astype(np.float32)
    if d.shape != (h, w): d = cv2.resize(d, (w, h))

    # 2. 严格同步的三维重投影与 Z-buffer 遮挡处理
    R = qvec2rotmat(qvec)
    sparse = np.zeros((h, w))
    mask = np.zeros((h, w), dtype=bool)
    pts_cam = pts_raw @ R.T + tvec # 转至相机系
    
    for i in range(len(pts_cam)):
        z = pts_cam[i, 2]
        if z <= DEPTH_MIN: continue
        u, v = int(xys_raw[i, 0]), int(xys_raw[i, 1])
        if 0 <= u < w and 0 <= v < h:
            # 仅保留视角内最近的深度点 (处理前景遮挡背景的情况)
            if sparse[v, u] == 0 or z < sparse[v, u]:
                sparse[v, u] = z
                mask[v, u] = True

    valid = mask & (sparse > 0)
    if np.sum(valid) < MIN_POINTS_GLOBAL: return None

    # 3. 执行对齐
    x_data, y_data = d[valid].reshape(-1, 1), sparse[valid]
    target_modes = ["linear","inverse","log","quadratic","inv_quad"] if mode == "auto" else [mode]

    best, stats = select_best(x_data, y_data, target_modes)
    if not best: return None

    model_g, mode_g = best
    aligned = apply_blocks(d, sparse, valid, model_g, mode_g, grid, overlap, local_fit, l_mode)
    aligned = np.clip(aligned, DEPTH_MIN, MAX_DEPTH)
    
    np.save(os.path.join(out_dir, base + ".npy"), aligned)
    return base, mode_g, stats

# ==============================================================================
# 【五、系统主入口与参数解析】
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MDE Scale 校正科研版 v4.3")

    # --- 基础路径参数 ---
    parser.add_argument("--colmap_dir", required=True, help="COLMAP sparse 目录路径")
    parser.add_argument("--npy_dir", required=True, help="MDE 原始深度 NPY 目录路径")
    parser.add_argument("--output_dir", required=True, help="输出结果存放路径")

    # --- 核心算法参数 ---
    parser.add_argument("--mode", default="auto", 
                        choices=["auto","linear","inverse","log","quadratic","inv_quad"],
                        help="对齐模式：auto 为自动选拔 RMSE 最优的模型")
    
    parser.add_argument("--grid", type=int, default=1, 
                        help="分块数量：1 代表全图对齐 (Baseline)；3 代表 3x3 分块")
    
    parser.add_argument("--overlap", type=float, default=0, 
                        help="块间重叠率：推荐 0.2-0.4，消除 3D 模型接缝的关键")

    # --- 局部精细化开关 ---
    parser.add_argument("--local_fit", type=str, default="True",
                        help="默认开启开启局部拟合。如需关闭，传入 False")
    
    parser.add_argument("--local_model_mode", default="global", 
                        choices=["global", "per_block"],
                        help="局部竞争策略：global (锁定全局最优名); per_block (每块独立选拔)")

    # --- 系统性能参数 ---
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")

    args = parser.parse_args()
    l_fit = True if args.local_fit.lower() == "true" else False

    # 逻辑保护：当不分块 (grid=1) 时，不需要执行局部拟合与重叠计算
    if args.grid == 1:
        l_fit = False
        args.overlap = 0

    # 创建输出目录
    out_npy_path = os.path.join(args.output_dir, "depth_aligned_npy")
    os.makedirs(out_npy_path, exist_ok=True)

    # 读取 COLMAP 重建数据
    # 自动检测 COLMAP 模型文件后缀
    ext = ".bin" if os.path.exists(os.path.join(args.colmap_dir, "images.bin")) else ".txt"
    cams, imgs, pts3D = read_model(args.colmap_dir, ext=ext)
    # cams, imgs, pts3D = read_model(args.colmap_dir)
    tasks = []
    
    # 构造数据并行任务池
    for k in imgs:
        img, cam = imgs[k], cams[imgs[k].camera_id]
        ids = img.point3D_ids
        v = ids != -1
        
        # 通过 Zip 同步遍历索引与坐标
        p3d_list = []
        xys_list = []
        for pid, xy in zip(ids[v], img.xys[v]):
            if pid in pts3D:
                p3d_list.append(pts3D[pid].xyz)
                xys_list.append(xy)
        
        if len(p3d_list) < MIN_POINTS_GLOBAL: continue

        # 打包单影像处理任务包
        tasks.append((
            img.name, img.qvec, img.tvec, cam.height, cam.width,
            np.array(p3d_list), np.array(xys_list), args.npy_dir, out_npy_path,
            args.mode, args.grid, args.overlap,
            l_fit, args.local_model_mode
        ))

    # 执行并行计算
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        for r in tqdm(exe.map(process, tasks), total=len(tasks), desc="🚀 校正进行中"):
            if r: results.append(r)

    # # 导出 CSV 对齐统计结果
    # with open(os.path.join(args.output_dir, "report.csv"), "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["image", "best_model_at_global"])
    #     for r in results: writer.writerow([r[0], r[1]])
    # ===== 导出详细报表 =====
    report_path = os.path.join(args.output_dir, "alignment_detailed_report.csv")
    
    # 定义所有参与消融竞争的模型列表
    m_list = ["linear", "inverse", "log", "quadratic", "inv_quad"]
    
    with open(report_path, "w", newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 1. 构造复杂的表头 (Header)
        # 包含：影像名、胜出模型、以及每个候选模型的 RMSE、内点率、内点数
        header = ["Image_Name", "Best_Model_Global"]
        for m in m_list:
            header += [f"{m}_RMSE", f"{m}_InlierRatio", f"{m}_InlierCount"]
        writer.writerow(header)

        # 2. 遍历结果并填充数据
        for res in results:
            # 过滤掉失败的任务
            if res is None or not isinstance(res, tuple): 
                continue
                
            base_name, win_mode, stats = res
            
            # 构造当前行数据
            row = [base_name, win_mode]
            for m in m_list:
                if m in stats:
                    s = stats[m]
                    row += [
                        f"{s['rmse']:.6f}", 
                        f"{s['inlier_ratio']:.4f}", 
                        s['num_inliers']
                    ]
                else:
                    # 如果该模型在拟合阶段失败（点数不足或不满足单调性），填入 N/A
                    row += ["N/A", "N/A", "N/A"]
            writer.writerow(row)

    print(f"详细报表已生成: {report_path}")
    print(f"\n校正任务已完成！\n结果存放于: {args.output_dir}")

if __name__ == "__main__":
    main()