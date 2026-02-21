# Matcha-AI-DTU Inference Engine (`services/inference`)

The pure Python powerhouse behind analyzing soccer videos within the Matcha-AI-DTU platform. Exposed out via **FastAPI** (`uvicorn`), this microservice digests videos forwarded by the Orchestrator, running state-of-the-art Computer Vision algorithms down to AI LLM audio synthesis.

## 🧠 Action Pipeline Overview
When a video hits the Inference API, it undergoes our sophisticated analysis pipeline (`app/core/analysis.py`):
1. **Ball Tracking & Goal Detection**: 
    - Custom CV utilizing `ultralytics` YOLOv8. 
    - Analyzes precise frames, utilizes bounding boxes with Intersection-over-Union mapping. 
    - `GoalDetectionEngine`: Specifically tailored to identify, track, auto-calibrate on center/goal lines, and confirm Goal occurrences!
2. **SoccerNet Action Parsing**:
    - Analyzes generic sports events: Tackles, Red cards, Offsides, Play action sequences.
3. **AI Commentary Synthesis**:
    - Integrates with Google's Language Models (`google-generativeai`). 
    - Given identified events, generates natural and enthusiastic play-by-play commentary scripts contextually.
4. **Text-To-Speech (TTS) Render**:
    - Consumes the AI generated scripts locally utilizing `piper-tts`.
    - Merges high-quality voiceover audio matching timestamps back onto the highlight reel seamlessly utilizing Torch and FFmpeg methodologies. 

## 🔌 Hardware Support Matrix
- CPU Support default natively.
- **CUDA 12.4 GPU acceleration** is heavily recommended for near real-time (5+ FPS) performance processing via `torch`/`torchvision` modules dynamically assigned in Python scripts.

## 🚀 Startup Environment

Due to heavy numerical/neural dependencies, creating an isolated virutal environment matching `python 3.11/3.12` is required. 

### Setting Requirements
```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate

# Install CUDA native builds safely first: (Recommended over native `pip install torch` locally on Windows with specific Cuda builds)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Execute Core packages
pip install -r requirements.txt
pip install piper-tts
```

### Configuration (`.env`)
You must define API Keys dynamically or as environment states.
- `ORCHESTRATOR_URL` = `http://localhost:4000` -> (Push updates)
- `GEMINI_API_KEY` = `your-api-key-here` -> (Context generation)

### Launch Service
```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Wait for Torch to load the YOLO pt weights into memory (usually local `.pt` caches).* 
API will open connections via `8000` and wait for tasks internally.

## 📊 Performance Notes
- Auto-rescaling implemented for larger streams protecting system VRAM limits natively.
- The `piper-tts` uses `onnxruntime` bindings natively sidestepping Visual C++ requirements.
- Goal detection pipeline successfully avoids heavy processing loops by selectively isolating frames down to key sequences saving CPU ticks overall.
