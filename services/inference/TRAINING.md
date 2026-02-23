# 🏗️ Training & Calibration Guide

This guide explains how to use the custom tools and frame-by-frame images to improve your sports analysis.

## 1. Fine-Tune YOLO (Object Detection)
**Goal**: Improve ball and player detection for your specific stadium/field.

1.  **Prepare Images**: Extract frames from your video.
2.  **Annotate**: Use [Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai) to draw boxes around `ball` and `player`. Export in **YOLO-v8** format.
3.  **Train**:
    ```bash
    cd services/inference
    ./venv/bin/python training/train_yolo.py --data /path/to/extracted/dataset/data.yaml --epochs 50
    ```
4.  **Deploy**: Once training finishes, copy the `best.pt` file to `services/inference/` and update `analysis.py` to use it:
    ```python
    model = YOLO("best.pt")
    ```

## 2. Pitch Calibration (Homography)
**Goal**: Transform "camera view" pixels into "real world" meters for exact ball speed and heatmap positioning.

1.  **Usage**:
    ```bash
    cd services/inference
    ./venv/bin/python tools/calibration/pitch_calibrator.py --image path/to/sample_frame.jpg
    ```
2.  **Instructions**:
    - A window will open. Click the 4 corners of the pitch or the penalty box in this order: **Top-Left, Top-Right, Bottom-Right, Bottom-Left**.
    - Press **'S'** to save.
3.  **Result**: This generates a `homography.json`. You can then replace the hardcoded `H_MATRIX` in `app/core/heatmap.py` with the matrix from this file for pixel-perfect speed calculations.

## 3. Custom Action Classifier (Action Spotting)
**Goal**: Detect specific events (headers, slide tackles, injury risk) using your specialized frames.

1.  **Prepare Data**: Organize your frames into subfolders named by the class:
    ```text
    /my_dataset
        /goal
        /tackle
        /injury_risk
        /header
    ```
2.  **Train**:
    ```bash
    cd services/inference
    ./venv/bin/python training/action_classifier.py --data /path/to/my_dataset --epochs 20
    ```
3.  **Result**: Saves a model at `models/action_classifier.pth`. You can integrate this into the main loop in `analysis.py` to classify every frame during the analysis phase.

---

### Tips for Better Results
- **Diversity**: Use images from different weather conditions (day/night/rain).
- **Quality**: Ensure your 4 calibration points are as accurate as possible. Even a 5-pixel error can lead to a 2km/h difference in ball speed.
- **Labeling**: Be consistent. If you label a "ball" as a "header" once, the model will get confused.
