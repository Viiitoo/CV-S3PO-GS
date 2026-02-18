#!/usr/bin/env python3
"""
生成手术器械 mask 脚本
针对 StereoMIS stereo_seq_easy 数据集，遮蔽右上方的 da Vinci 手术器械

器械外观：
- 蓝灰色圆柱形手臂 (da Vinci 机器人臂，带 "da Vinci Surgery" 字样)
- 银色金属轴杆 (后期帧，位于右侧边缘)

策略：在图像上半部分+右侧边缘的 ROI 内，检测非组织像素
"""

import cv2
import numpy as np
import os
from glob import glob

RGB_DIR = "datasets/StereoMIS/stereo_seq_easy/stereo_seq_easy/train/rgb"
MASK_DIR = "datasets/StereoMIS/stereo_seq_easy/stereo_seq_easy/train/instrument_mask"


def create_instrument_mask(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ============================================================
    # ROI 定义
    # ============================================================
    roi_top   = np.zeros((h, w), dtype=np.uint8)
    roi_top[:int(h * 0.24), :]                  = 255   # 顶部 24%（蓝臂/把手）
    roi_top[:int(h * 0.12), int(w * 0.28):]     = 255   # 顶右延伸

    roi_right = np.zeros((h, w), dtype=np.uint8)
    roi_right[:, int(w * 0.70):]                = 255   # 右侧 30%（金属轴）

    # ============================================================
    # 1. 蓝灰色 da Vinci 手臂 (H=95-140, S=15-150, V=45-240)
    #    仅在顶部 ROI 内检测
    # ============================================================
    blue_arm = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([95, 15, 45]), np.array([140, 150, 240])),
        roi_top
    )

    # ============================================================
    # 2. 金属轴杆 — 分两部分检测，仅在右侧 ROI 内
    # ============================================================
    # 2a. 高光中心: 近白色高亮 (S<22, V>195) — 轴杆镜面反射
    shaft_highlight = cv2.inRange(hsv,
                                  np.array([0,   0, 195]),
                                  np.array([180, 22, 255]))

    # 2b. 暗环/轴体: 冷色调暗红 (H=155-179, S=55-210, V=45-175)
    #    与组织(H=0-20 暖橙)明显区分
    shaft_body = cv2.inRange(hsv,
                             np.array([155, 55, 45]),
                             np.array([179, 210, 175]))

    metal_shaft = cv2.bitwise_and(
        cv2.bitwise_or(shaft_highlight, shaft_body),
        roi_right
    )

    combined = cv2.bitwise_or(blue_arm, metal_shaft)

    # ============================================================
    # 3. 排除极暗区域 (腹腔镜视场黑边, V<40)
    # ============================================================
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(dark))

    # ============================================================
    # 4. 形态学：填洞连通 + 轻度膨胀以覆盖器械边缘
    # ============================================================
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5,  iterations=5)
    combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, k11, iterations=2)

    # ============================================================
    # 5. 去除过小噪点 (< 400 像素)
    # ============================================================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    clean = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            clean[labels == i] = 255

    return clean


def main():
    os.makedirs(MASK_DIR, exist_ok=True)

    img_paths = sorted(glob(os.path.join(RGB_DIR, "*.png")))
    print(f"共找到 {len(img_paths)} 张图像")

    for idx, img_path in enumerate(img_paths):
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [!] 读取失败: {img_path}")
            continue

        mask = create_instrument_mask(img)
        out_path = os.path.join(MASK_DIR, fname)
        cv2.imwrite(out_path, mask)

        if idx % 20 == 0:
            print(f"  [{idx:03d}/{len(img_paths)}] {fname} -> 器械像素数: {mask.sum()//255}")

    print(f"\n完成！mask 保存至: {MASK_DIR}")


if __name__ == "__main__":
    main()
