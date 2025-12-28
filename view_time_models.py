#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速查看各个时刻的模型
使用方法:
    python view_time_models.py <save_dir> [--frame <frame_idx>] [--frames <frame1> <frame2> ...] [--animate]
"""
import open3d as o3d
import os
import sys
import argparse
import numpy as np

def find_time_dir(save_dir):
    """查找time目录"""
    possible_paths = [
        os.path.join(save_dir, "point_cloud", "final", "time"),
        os.path.join(save_dir, "point_cloud", "final_after_opt", "time"),
        os.path.join(save_dir, "point_cloud", "time"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def view_single_frame(time_dir, frame_idx):
    """查看单个时刻的模型"""
    ply_path = os.path.join(time_dir, "time_{:04d}.ply".format(frame_idx))
    
    if not os.path.exists(ply_path):
        print("Error: File not found: {}".format(ply_path))
        return False
    
    print("Loading: {}".format(ply_path))
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) == 0:
        print("Error: Point cloud is empty")
        return False
    
    print("Points: {:,}".format(len(pcd.points)))
    print("Colors: {}".format("Yes" if len(pcd.colors) > 0 else "No"))
    
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name="Time Step {}".format(frame_idx),
        width=1024,
        height=768
    )
    return True

def view_multiple_frames(time_dir, frame_indices):
    """查看多个时刻的模型（对比）"""
    pcds = []
    colors = [
        [1, 0, 0],    # 红
        [0, 1, 0],    # 绿
        [0, 0, 1],    # 蓝
        [1, 1, 0],    # 黄
        [1, 0, 1],    # 紫
        [0, 1, 1],    # 青
        [1, 0.5, 0],  # 橙
        [0.5, 0, 1],  # 紫蓝
    ]
    
    print("Loading {} time-step models...".format(len(frame_indices)))
    
    for i, frame_idx in enumerate(frame_indices):
        ply_path = os.path.join(time_dir, "time_{:04d}.ply".format(frame_idx))
        
        if not os.path.exists(ply_path):
            print("Warning: Skipping {} (not found)".format(ply_path))
            continue
        
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) == 0:
            print("Warning: Skipping frame {} (empty point cloud)".format(frame_idx))
            continue
        
        pcd.paint_uniform_color(colors[i % len(colors)])
        pcds.append(pcd)
        print("Frame {}: {:,} points".format(frame_idx, len(pcd.points)))
    
    if not pcds:
        print("Error: No available point clouds")
        return False
    
    print("\nDisplaying {} models (different colors)...".format(len(pcds)))
    o3d.visualization.draw_geometries(
        pcds, 
        window_name="Multiple Time Steps Comparison",
        width=1024,
        height=768
    )
    return True

def list_available_frames(time_dir):
    """列出所有可用的帧"""
    if not os.path.exists(time_dir):
        print("Error: Directory not found: {}".format(time_dir))
        return []
    
    ply_files = sorted([f for f in os.listdir(time_dir) if f.endswith('.ply') and f.startswith('time_')])
    
    if not ply_files:
        print("Error: No time_*.ply files found in: {}".format(time_dir))
        return []
    
    frame_indices = []
    for f in ply_files:
        try:
            frame_idx = int(f.split('_')[1].split('.')[0])
            frame_indices.append(frame_idx)
        except:
            continue
    
    return sorted(frame_indices)

def animate_frames(time_dir, start=0, end=None, step=1, delay=0.1):
    """动画显示各个时刻"""
    import time
    
    frame_indices = list_available_frames(time_dir)
    
    if not frame_indices:
        return False
    
    if end is None:
        end = max(frame_indices)
    
    # 过滤帧范围
    frame_indices = [f for f in frame_indices if start <= f <= end]
    frame_indices = frame_indices[::step]  # 按步长采样
    
    if not frame_indices:
        print("Error: No frames found in range [{}, {}]".format(start, end))
        return False
    
    print("Animation mode: {} frames (step={})".format(len(frame_indices), step))
    print("Tip: Close window or press ESC to exit")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("Time Deformation Animation", width=1024, height=768)
    
    # 加载第一帧
    first_frame = frame_indices[0]
    ply_path = os.path.join(time_dir, "time_{:04d}.ply".format(first_frame))
    pcd = o3d.io.read_point_cloud(ply_path)
    vis.add_geometry(pcd)
    
    current_idx = 0
    while True:
        # 更新到下一帧
        current_idx = (current_idx + 1) % len(frame_indices)
        frame_idx = frame_indices[current_idx]
        
        ply_path = os.path.join(time_dir, "time_{:04d}.ply".format(frame_idx))
        new_pcd = o3d.io.read_point_cloud(ply_path)
        
        vis.remove_geometry(pcd, reset_bounding_box=False)
        vis.add_geometry(new_pcd, reset_bounding_box=False)
        pcd = new_pcd
        
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(delay)
        
        # 检查窗口是否关闭
        if not vis.poll_events():
            break
    
    vis.destroy_window()
    return True

def compare_with_canonical(time_dir, frame_idx, canonical_path=None):
    """对比canonical模型和指定时刻的模型"""
    # 查找canonical模型
    if canonical_path is None:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(time_dir)), "point_cloud.ply"),
            os.path.join(os.path.dirname(time_dir), "point_cloud.ply"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                canonical_path = path
                break
    
    if canonical_path is None or not os.path.exists(canonical_path):
        print("Warning: Canonical model not found, showing only deformed model")
        return view_single_frame(time_dir, frame_idx)
    
    # 加载两个模型
    pcd_canonical = o3d.io.read_point_cloud(canonical_path)
    ply_path = os.path.join(time_dir, "time_{:04d}.ply".format(frame_idx))
    pcd_deformed = o3d.io.read_point_cloud(ply_path)
    
    # 设置颜色
    pcd_canonical.paint_uniform_color([1, 0, 0])  # 红色 - 原始
    pcd_deformed.paint_uniform_color([0, 1, 0])   # 绿色 - 形变后
    
    print("Canonical: {:,} points (red)".format(len(pcd_canonical.points)))
    print("Deformed (t={}): {:,} points (green)".format(frame_idx, len(pcd_deformed.points)))
    
    # 计算位移统计
    if len(pcd_canonical.points) == len(pcd_deformed.points):
        points_canonical = np.asarray(pcd_canonical.points)
        points_deformed = np.asarray(pcd_deformed.points)
        displacements = np.linalg.norm(points_deformed - points_canonical, axis=1)
        print("Displacement stats: mean={:.6f}, max={:.6f}, std={:.6f}".format(
            displacements.mean(), displacements.max(), displacements.std()))
    
    o3d.visualization.draw_geometries(
        [pcd_canonical, pcd_deformed],
        window_name="Canonical vs Frame {}".format(frame_idx),
        width=1024,
        height=768
    )
    return True

def main():
    parser = argparse.ArgumentParser(
        description="View time-step models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View single time step
  python view_time_models.py results/.../2025-12-27-19-11-19 --frame 50
  
  # Compare multiple time steps
  python view_time_models.py results/.../2025-12-27-19-11-19 --frames 0 25 50 75 99
  
  # Animate through all frames
  python view_time_models.py results/.../2025-12-27-19-11-19 --animate --start 0 --end 99 --step 5
  
  # Compare with canonical model
  python view_time_models.py results/.../2025-12-27-19-11-19 --frame 50 --compare
        """
    )
    
    parser.add_argument("save_dir", type=str, help="Result save directory (contains point_cloud)")
    parser.add_argument("--frame", type=int, help="View single time step")
    parser.add_argument("--frames", nargs="+", type=int, help="View multiple time steps (compare)")
    parser.add_argument("--animate", action="store_true", help="Animate through all frames")
    parser.add_argument("--start", type=int, default=0, help="Start frame for animation")
    parser.add_argument("--end", type=int, default=None, help="End frame for animation")
    parser.add_argument("--step", type=int, default=1, help="Step size for animation")
    parser.add_argument("--delay", type=float, default=0.1, help="Animation delay (seconds)")
    parser.add_argument("--compare", action="store_true", help="Compare with canonical model (requires --frame)")
    parser.add_argument("--list", action="store_true", help="List all available frames")
    
    args = parser.parse_args()
    
    # 查找time目录
    time_dir = find_time_dir(args.save_dir)
    
    if time_dir is None:
        print("Error: Time directory not found")
        print("  Search path: {}".format(args.save_dir))
        print("  Possible reasons:")
        print("    1. Training not completed yet")
        print("    2. Model does not contain deformation parameters")
        print("    3. Incorrect save path")
        return
    
    print("Found time directory: {}".format(time_dir))
    
    # 列出所有帧
    if args.list:
        frame_indices = list_available_frames(time_dir)
        if frame_indices:
            print("\nAvailable frames ({}):".format(len(frame_indices)))
            print("  Range: {} - {}".format(min(frame_indices), max(frame_indices)))
            sample_indices = frame_indices[::max(1, len(frame_indices)//10)]
            print("  Sample: {}".format(sample_indices))
        return
    
    # 执行不同的查看模式
    if args.frame is not None:
        if args.compare:
            compare_with_canonical(time_dir, args.frame)
        else:
            view_single_frame(time_dir, args.frame)
    elif args.frames:
        view_multiple_frames(time_dir, args.frames)
    elif args.animate:
        animate_frames(time_dir, args.start, args.end, args.step, args.delay)
    else:
        # 默认：列出可用帧并询问
        frame_indices = list_available_frames(time_dir)
        if frame_indices:
            print("\nAvailable frames: {} (range: {} - {})".format(
                len(frame_indices), min(frame_indices), max(frame_indices)))
            print("\nUsage:")
            print("  python view_time_models.py {} --frame <frame_idx>".format(args.save_dir))
            print("  python view_time_models.py {} --frames 0 25 50 75 99".format(args.save_dir))
            print("  python view_time_models.py {} --animate".format(args.save_dir))
            print("  python view_time_models.py {} --list".format(args.save_dir))
        else:
            print("Error: No available frames found")

if __name__ == "__main__":
    main()
