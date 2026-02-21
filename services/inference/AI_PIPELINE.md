# AI Pipeline: The 4-Phase Architecture

Matcha-AI-DTU's Inference module runs on a highly structured 4-phase sequential pipeline. This document breaks down the operations occurring when the Orchestrator delegates a video to `services/inference/app/core/analysis.py`.

## 🔄 The Pipeline Architecture

```mermaid
graph TD
    A[Raw Video] --> B{Phase 1a: Prep}
    B --> C[Phase 1b: Goal Detection Pipeline]
    C --> D[Phase 2: Action Recognition]
    D --> E[Phase 3: Contextual AI Commentary]
    E --> F[Phase 4: Synthesis & Output]
    F --> G[Processed Video & TTS Audio]
    
    subgraph Phase 1b (YOLOv8 + BBox Tracing)
    C1(Find Ball) --> C2(Calculate IoU/Movement)
    C2 --> C3(Detect Goal Line Intersect)
    C3 --> C4(Confirm Goal event)
    end
    C --> Phase 1b
```

### 🎯 Phase 1a & 1b: Vision & Goal Detection (YOLOv8)
The video is initially parsed frame-by-frame via `cv2.VideoCapture`. 
To ensure realtime performance, we process frames in intervals (configurable via `analysis.py` constants, generally bypassing 4 frames for every 1 processed).

1. **Object Detection**: YOLOv8 models (`yolov8n.pt` or `yolov8s.pt`) are executed per frame to isolate the sports ball.
2. **IoU Tracking**: The `BallTracker` module maps bounding boxes across sequential frames using Intersection-over-Union, retaining ball trajectory even if YOLO misses a detection for 1-2 frames due to occlusion.
3. **Goal Detection Engine**: 
   - `GoalLineCalibrator`: Scans the field to establish goal line boundaries (either automatically or manually via config overrides).
   - If the tracked ball's coordinates cross the calculated threshold for > `X` consecutive frames, a `GOAL` event is confidently emitted.

### 🏃 Phase 2: Action Parsing (Decision Tree Logic)
While Goal Detection zeroes in on scoring, Phase 2 categorizes broader events:
- **Tackles, Fouls, Celebrations, Cards**.
- Action segments are matched against timestamp matrices. We construct an array of `Event` objects holding the `{ type, timestamp, confidence_score }`.

### 🧠 Phase 3: Generative Commentary (Google Gemini)
With our list of `Events` populated, we need human-like commentary to match.
1. A structured prompt is built feeding Gemini the parsed actions. Example:
   *"At 0:15, a foul occurred. At 0:42, a GOAL was scored! Act like an enthusiastic sports commentator and write a short script..."*
2. Gemini returns an animated, contextual string.

### 🎙️ Phase 4: Audio Synthesis (Piper TTS)
Finally, we bridge the generated script back into reality:
1. `piper-tts` loads the lightweight `en_US-lessac-medium.onnx` model into memory.
2. The Gemini script is synthesized into a `.wav` file entirely offline on the machine.
3. (Optional step): `FFmpeg` can then stitch this generated `.wav` back onto the original `mp4` asset to serve a final highlight reel to the frontend.

## ⚙️ Customizing the Pipeline
Developers modifying `app/core/analysis.py` should focus on the `CONFIG` dict:
```python
CONFIG = {
    "GOAL_DETECTION_MIN_FRAMES": 3, # Lower for sensitivity, raise for accuracy
    "MIN_CONFIDENCE": 0.55          # YOLO confidence threshold
}
```
