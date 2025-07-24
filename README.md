# 🚗 Vehicle Attribute Extraction from Remote Sensing Imagery

Author: Jun Xiong <junxiong360@gmail.com>

This project processes vehicle detections (in oriented bounding box format) from 30cm GSD remote sensing images to extract patch crops, estimate dominant colors, and build an interactive sample selection tool to support vehicle type labeling and model training.

---

## 📁 Project Structure

```

vehicle_patches/
│   ├── 001_0001.jpg         # Cropped OBB vehicle patch (RGB)
│   ├── vehicle_meta.csv     # Metadata for each patch (bbox, angle, class)
│   └── vehicle_color_labeled.csv  # Includes color category (e.g. Red, White)
data/
│   ├── 1.png                # Raw aerial image
│   ├── 1.txt                # Corresponding annotation in LE90 format
scripts/
│   ├── crop_obb_patches.py
│   ├── classify_color_by_hsv.py
│   └── sample_umap_picker.ipynb

```

---

## 🔧 Scripts Overview

### 1. `crop_obb_patches.py`

> Extracts rotated vehicle image patches from LE90-annotated bounding boxes.

- Input: `.png` images and `.txt` annotation files in LE90 format:
```

<class> \<x\_center> \<y\_center> <width> <height> <theta>

````
- Output:
- Rotated and cropped vehicle patches in `vehicle_patches/`
- Metadata CSV: `vehicle_meta.csv` with bbox geometry and class ID

- Key functions:
- `obb2poly_le90`: Converts LE90 format to polygon
- `crop_rotated_patch`: Applies perspective transform to extract patches

---

### 2. `classify_color_by_hsv.py`

> Estimates the dominant color for each patch using HSV-based rule heuristics.

- Input: `vehicle_meta.csv` + cropped patch images
- Output: `vehicle_color_labeled.csv` with added `color` field
- Categories include: `Black`, `White`, `Gray`, `Silver`, `Blue`, `Red`, `Brown`, `Gold`, `Green`, `Tan`, `Orange`, `Yellow`

- Approach:
- Converts image to HSV
- Finds mode (dominant) HSV value
- Maps it to pre-defined color categories via threshold rules

---

### 3. `sample_picker_umap.py`

> Interactive UMAP-based visualizer for patch exploration and labeling support.

- Input: `vehicle_color_labeled.csv` and image files
- Output: Interactive Bokeh plot in Jupyter Notebook

- Key features:
- Extracts HOG features from image patches
- Projects to 2D using `UMAP`
- Generates image thumbnails as hover tooltips (RGB)
- Groups visually similar vehicles to aid manual labeling

---

## ✅ Requirements

Install dependencies with:

```bash
uv pip install opencv-python numpy pandas pillow scikit-image umap-learn scikit-learn bokeh
````

---

## 🧠 Next Steps

* [ ] Manually label 50–100 patches per vehicle type
* [ ] Train SVM / XGBoost classifier for vehicle type inference
* [ ] Integrate `shadow_ratio` and `occlusion_ratio` estimation modules
* [ ] Deploy patch labeling UI or use `Gradio`/`Streamlit` for rapid annotation