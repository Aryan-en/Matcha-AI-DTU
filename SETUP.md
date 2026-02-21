# Matcha-AI-DTU Setup Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | 18+ | Frontend & Orchestrator |
| Python | 3.11 | Inference service (TTS requires 3.11/3.12) |
| Docker Desktop | Latest | PostgreSQL + Redis |
| FFmpeg | Latest | Video processing |
| NVIDIA GPU | CUDA 12.4 | GPU acceleration (optional) |

---

## Quick Start

### 1. Start Infrastructure (Docker)

```powershell
cd Matcha-AI-DTU
docker compose up -d
```

This starts:
- **PostgreSQL** on port 5433
- **Redis** on port 6380

---

### 2. Install Dependencies

```powershell
# Root workspace
npm install

# Frontend
cd apps/web
npm install

# Orchestrator
cd ../../services/orchestrator
npm install
npx prisma generate
npx prisma migrate deploy
```

---

### 3. Python Environment (Inference)

```powershell
cd services/inference

# Create Python 3.11 virtual environment
py -3.11 -m venv venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt

# Install TTS (Piper)
pip install piper-tts
```

---

### 4. Start Services

Open 4 terminals:

**Terminal 1 - Docker (already running)**
```powershell
docker compose up -d
```

**Terminal 2 - Orchestrator (port 4000)**
```powershell
cd services/orchestrator
npm run start:dev
```

**Terminal 3 - Inference (port 8000)**
```powershell
cd services/inference
$env:ORCHESTRATOR_URL="http://localhost:4000"
$env:GEMINI_API_KEY="your-api-key-here"
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 4 - Frontend (port 3000)**
```powershell
cd apps/web
npm run dev
```

---

### 5. Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Orchestrator API | http://localhost:4000 |
| Inference API | http://localhost:8000 |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   Next.js   │────▶│   NestJS     │────▶│   FastAPI     │
│   Frontend  │     │ Orchestrator │     │   Inference   │
│   :3000     │     │    :4000     │     │     :8000     │
└─────────────┘     └──────────────┘     └───────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌────────────┐           ┌────────────┐
       │ PostgreSQL │           │   Redis    │
       │   :5433    │           │   :6380    │
       └────────────┘           └────────────┘
```

---

## TTS: Piper TTS

### Why Piper TTS (not Coqui)?

| Criteria | Coqui TTS | Piper TTS |
|----------|-----------|-----------|
| **Windows Install** | ❌ Fails - requires Visual C++ Build Tools | ✅ Pre-built wheels |
| **numpy compatibility** | ❌ Builds numpy 1.22 from source | ✅ Works with numpy 1.26+ |
| **Dependencies** | Complex, compilation required | Minimal (onnxruntime) |
| **Quality** | High | High (neural ONNX models) |

### Failed Coqui Attempts

```powershell
# All failed:
pip install TTS                      # wheel build error
pip install TTS==0.17.4              # numpy build needs Visual C++
pip install TTS --no-build-isolation # still fails
```

### Piper TTS Solution

```powershell
pip install piper-tts  # Works immediately
```

**Voice Model**: `en_US-lessac-medium` (downloaded automatically from HuggingFace on first run)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_URL` | http://localhost:4000 | Orchestrator endpoint |
| `GEMINI_API_KEY` | - | Google Gemini API key for AI commentary |
| `INFERENCE_URL` | http://localhost:8000 | Inference service endpoint |

---

## GPU Support

The inference service uses **CUDA 12.4** for GPU acceleration:

```python
import torch
print(torch.cuda.is_available())  # True if GPU works
print(torch.cuda.get_device_name(0))  # e.g., "NVIDIA GeForce RTX 3050"
```

---

## Troubleshooting

### Port in use
```powershell
# Check what's using a port
netstat -ano | Select-String ":3000"

# Kill process by PID
Stop-Process -Id <PID> -Force
```

### Docker container conflict
```powershell
docker ps -a  # Find conflicting containers
docker stop <container_name>
```

### Python import errors
```powershell
# Activate the venv first!
.\venv\Scripts\activate

# Re-install dependencies
pip install -r requirements.txt
pip install piper-tts
```

---

## File Structure

```
Matcha-AI-DTU/
├── apps/
│   └── web/              # Next.js frontend
├── services/
│   ├── orchestrator/     # NestJS API + WebSocket
│   └── inference/        # FastAPI + YOLO + TTS
│       ├── venv/         # Python 3.11 virtualenv
│       ├── models/       # Downloaded TTS models
│       └── uploads/      # Uploaded videos
├── uploads/              # Shared upload directory
└── docker-compose.yml    # PostgreSQL + Redis
```
