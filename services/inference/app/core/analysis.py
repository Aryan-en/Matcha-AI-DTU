import cv2
import logging
import requests
import os
import torch
from pathlib import Path
from ultralytics import YOLO

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Workaround for Torch 2.4+ security feature
try:
    from ultralytics.nn.tasks import DetectionModel
    # Check if add_safe_globals exists (added in newer torch)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    logger.warning(f"Could not add safe globals: {e}")

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://host.docker.internal:4000")

# Load model (will download on first run)
model = YOLO('yolov8n.pt') 

def analyze_video(video_path: str, match_id: str):
    logger.info(f"Starting analysis for match {match_id} on video {video_path}")
    
    # In a real scenario, download video from S3 if video_path is a URL
    # For MVP, we assume it might be accessible or we mock the processing
    
    results_summary = []
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {"error": "Could not open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        events = []

        # Process every Nth frame to speed up MVP
        skip_frames = 30 
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % skip_frames == 0:
                # Calculate and report progress
                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    try:
                        requests.post(
                            f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                            json={"progress": progress},
                            timeout=1
                        )
                    except Exception as e:
                        logger.warning(f"Failed to report progress: {e}")

                # Run YOLO inference
                results = model(frame, verbose=False)
                
                # Extract detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        
                        # Basic logic: Detect 'sports ball' or 'person'
                        # In real app, we'd fine-tune this for goals/fouls
                        if label in ['sports ball', 'person'] and conf > 0.5:
                             events.append({
                                 "timestamp": frame_count / 30.0, # assuming 30fps
                                 "type": label,
                                 "confidence": conf
                             })

            frame_count += 1

        cap.release()
        
        logger.info(f"Analysis complete. Found {len(events)} events.")
        return {"match_id": match_id, "events": events, "status": "completed"}

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {"error": str(e)}
