import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# Parameters
IMG_DIR = Path("data")
PATCH_DIR = Path("vehicle_patches")
PATCH_DIR.mkdir(exist_ok=True)
IMG_SIZE = 1024
PATCH_SIZE = 128

# Helper to convert LE90 OBB to polygon (based on mmrotate)
def obb2poly_le90(x_c, y_c, w, h, theta_deg):
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = w / 2
    dy = h / 2
    # Define rectangle corners centered at (0,0)
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])
    # Rotate and translate
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = corners @ rot.T
    translated = rotated + np.array([x_c, y_c])
    return translated

# Read annotations and crop rotated patches
records = []

for i in range(1, 46):
    img_path = IMG_DIR / f"{i}.png"
    txt_path = IMG_DIR / f"{i}.txt"

    if not img_path.exists() or not txt_path.exists():
        continue

    img = cv2.imread(str(img_path))
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for j, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls, x_c, y_c, w, h, theta = map(float, parts)
        # De-normalize
        x_c *= IMG_SIZE
        y_c *= IMG_SIZE
        w *= IMG_SIZE
        h *= IMG_SIZE

        poly = obb2poly_le90(x_c, y_c, w, h, theta)

        # Define destination square
        dst_pts = np.array([
            [0, 0],
            [PATCH_SIZE-1, 0],
            [PATCH_SIZE-1, PATCH_SIZE-1],
            [0, PATCH_SIZE-1]
        ], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(np.float32(poly), dst_pts)
            warped = cv2.warpPerspective(img, M, (PATCH_SIZE, PATCH_SIZE))
            out_name = f"{i:03d}_{j:04d}.jpg"
            out_path = PATCH_DIR / out_name
            cv2.imwrite(str(out_path), warped)

            records.append({
                "vehicle_id": f"{i:03d}_{j:04d}",
                "image_file": out_name,
                "source_image": f"{i}.png",
                "x_center": x_c,
                "y_center": y_c,
                "width": w,
                "height": h,
                "theta_deg": theta,
                "class_id": int(cls)
            })
        except:
            continue

# Save metadata CSV
df = pd.DataFrame(records)
df.to_csv(PATCH_DIR / "vehicle_meta.csv", index=False)
