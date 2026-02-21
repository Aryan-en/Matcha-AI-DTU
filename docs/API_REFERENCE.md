# 🔌 API & Events Reference Guide

This document outlines the contracts established between Next.js $\rightarrow$ Orchestrator $\leftrightarrow$ Inference Engine.

---

## 1. Orchestrator API (NestJS - Port 4000)

These are operations exposed primarily to the Frontend Client (Next.js).

### `POST /matches`
Initiates a new video processing task.
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: `(Binary)` The video file (e.g. .mp4).
  - `title`: `(String)` Optional.
- **Response**: `201 Created`
  ```json
  {
      "id": "match_123xyz",
      "status": "PROCESSING",
      "videoUrl": "/uploads/raw_video.mp4"
  }
  ```

### `GET /matches/:id`
Retrieves analyzed data for a specific match.
- **Response**: `200 OK`
  ```json
  {
      "id": "match_123xyz",
      "status": "COMPLETED",
      "videoUrl": "/uploads/raw_video.mp4",
      "ttsAudioUrl": "/uploads/tts_output.wav",
      "events": [
          {
              "type": "GOAL",
              "timestampSeconds": 42.5,
              "confidence": 0.98,
              "description": "Unbelievable strike hitting the top corner!"
          }
      ]
  }
  ```

---

## 2. Orchestrator WebSocket Events (Socket.IO)

Clients must connect to `ws://localhost:4000`.

### Event `analysisProgress`
Emitted continously as Python YOLO evaluates the video framework.
```json
{
    "matchId": "match_123xyz",
    "percentage": 45.5,
    "statusMessage": "Tracking ball across sector 4..."
}
```

### Event `eventDetected`
Emitted immediately when Phase 1b (Goal Detection) or Phase 2 catches an activity.
```json
{
    "matchId": "match_123xyz",
    "eventType": "GOAL",
    "timestamp": 42.5,
    "confidence": 0.98
}
```

### Event `analysisComplete`
Emitted when Phase 4 (TTS generation) is completed.
```json
{
    "matchId": "match_123xyz",
    "finalAudioUrl": "/uploads/tts_output.wav"
}
```

---

## 3. Inference Engine API (FastAPI - Port 8000)

Operations exposed natively *(Orchestrator $\rightarrow$ Python)*.

### `POST /analyze`
Triggers the inference processing queue asynchronously using FastAPI BackgroundTasks.
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
      "matchId": "match_123xyz",
      "videoFilePath": "C:/Projects/Matcha-AI/uploads/raw_video.mp4",
      "callbackUrl": "http://localhost:4000/callbacks"
  }
  ```
- **Response**: `202 Accepted` (Worker started)

---

## 4. Orchestrator Internal Callbacks (Port 4000)

Private routes utilized by the Python script to push updates back to NestJS.

### `POST /callbacks/progress`
- **Body**: `{ "matchId": "...", "percentage": 45.5 }`

### `POST /callbacks/event`
- **Body**: `{ "matchId": "...", "event": { /* event struct */ }}`

### `POST /callbacks/complete`
- **Body**: `{ "matchId": "...", "ttsPath": "/path/to/wav" }`
