import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Patch and output path
PATCH_DIR = Path("vehicle_patches")
META_PATH = PATCH_DIR / "vehicle_meta.csv"
OUT_CSV = PATCH_DIR / "vehicle_color_labeled.csv"

# Load metadata
df = pd.read_csv(META_PATH)

# Color label mapping rules (Hue range in OpenCV: 0-179)
COLOR_RULES = {
    "Red":      lambda h, s, v: ((h <= 10 or h >= 160) and s > 50 and v > 50),
    "Orange":   lambda h, s, v: (11 <= h <= 25 and s > 50 and v > 50),
    "Yellow":   lambda h, s, v: (26 <= h <= 34 and s > 50 and v > 50),
    "Green":    lambda h, s, v: (35 <= h <= 85 and s > 50 and v > 50),
    "Blue":     lambda h, s, v: (86 <= h <= 130 and s > 50 and v > 50),
    "Brown":    lambda h, s, v: (10 <= h <= 30 and s > 50 and v < 80),
    "Gray":     lambda h, s, v: (s < 30 and 50 <= v <= 200),
    "Black":    lambda h, s, v: (v < 50),
    "White":    lambda h, s, v: (v > 200 and s < 30),
    "Silver":   lambda h, s, v: (s < 50 and 150 < v <= 200),
    "Tan":      lambda h, s, v: (20 <= h <= 35 and s < 60 and 120 <= v <= 200),
    "Gold":     lambda h, s, v: (20 <= h <= 35 and s > 60 and v > 150),
}

def classify_color_by_mode(hsv_img):
    h, s, v = cv2.split(hsv_img)
    h_flat, s_flat, v_flat = h.flatten(), s.flatten(), v.flatten()

    # Use mode of pixels to determine dominant tone
    pixels = zip(h_flat, s_flat, v_flat)
    mode_pixel = Counter(pixels).most_common(1)[0][0]
    h_mode, s_mode, v_mode = mode_pixel

    for color, rule in COLOR_RULES.items():
        if rule(h_mode, s_mode, v_mode):
            return color
    return "Unknown"

# Process each patch image
color_labels = []
for idx, row in df.iterrows():
    img_path = PATCH_DIR / row["image_file"]
    if not img_path.exists():
        color_labels.append("Missing")
        continue

    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    label = classify_color_by_mode(hsv)
    color_labels.append(label)

df["color"] = color_labels
df.to_csv(OUT_CSV, index=False)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Color-Labeled Vehicle Metadata", dataframe=df)
