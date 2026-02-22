# Matcha-AI-DTU Setup Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | 18+ | Frontend & Orchestrator |
| Python | 3.11+ | Inference service (3.14 compatible) |
| Docker Desktop | Latest | PostgreSQL + Redis |
| FFmpeg | Latest | Video processing & highlight reel |
| NVIDIA GPU | CUDA 12.4 | GPU acceleration (optional but recommended) |

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

### 2. Install Node Dependencies

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

# Create Python virtual environment
py -3.11 -m venv venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support (if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all requirements (includes lapx, huggingface-hub, edge-tts)
pip install -r requirements.txt
```

> ✅ **`piper-tts` is no longer required.** The TTS system has been upgraded to Kokoro-82M (HuggingFace) → edge-tts → silent fallback. See the TTS section below.

---

### 4. Configure Environment Variables

**`services/orchestrator/.env`** (create this file):
```env
DATABASE_URL="postgresql://matcha_user:matcha_password@localhost:5433/matcha_db?schema=public"
HF_TOKEN=hf_your_huggingface_token_here
CORS_ORIGIN=http://localhost:3000
INFERENCE_URL=http://localhost:8000
PORT=4000
```

**`services/inference/.env`** (create this file):
```env
GEMINI_API_KEY=your_google_ai_studio_api_key
HF_TOKEN=hf_your_huggingface_token_here
ORCHESTRATOR_URL=http://localhost:4000
```

> 🔑 Gemini API key: [Google AI Studio](https://makersuite.google.com/app/apikey)  
> 🔑 HuggingFace token (free, "Read" scope): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
> `HF_TOKEN` is optional but highly recommended for Kokoro TTS (Tier 1).

---

### 5. Start Services

**Recommended — single command with Turborepo:**
```powershell
# From the monorepo root
npx turbo run dev
```

**Or manually in 4 terminals:**

**Terminal 1 — Docker (already running)**
```powershell
docker compose up -d
```

**Terminal 2 — Orchestrator (port 4000)**
```powershell
cd services/orchestrator
npm run start:dev
```

**Terminal 3 — Inference (port 8000)**
```powershell
cd services/inference
.\venv\Scripts\activate
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```
> The `.env` file in `services/inference/` is loaded automatically by Python's `os.getenv()`.

**Terminal 4 — Frontend (port 3000)**
```powershell
cd apps/web
npm run dev
```

---

### 6. Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Orchestrator API | http://localhost:4000 |
| Inference API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |

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

## TTS: 3-Tier Neural Voice System (Updated)

The TTS system has been upgraded from `piper-tts` to a 3-tier cascade for maximum quality and reliability:

| Tier | Model | Quality | Requirement |
|------|-------|---------|-------------|
| 🥇 1 | **Kokoro-82M** (`hexgrad/Kokoro-82M`) | Ultra-high — #1 TTS Arena | `HF_TOKEN` env var (free) |
| 🥈 2 | **Microsoft edge-tts** (`en-GB-RyanNeural`) | High — neural, no key needed | `edge-tts` pip package |
| 🥉 3 | **FFmpeg silence** | — | FFmpeg in PATH |

The system tries Tier 1 → falls back to Tier 2 → falls back to Tier 3 automatically.

### Why we moved away from Piper TTS

| Criteria | Piper TTS (old) | Kokoro-82M (new) |
|----------|----------------|-----------------|
| **Quality** | Good | Ultra-high (#1 arena ranking) |
| **Windows install** | ✅ Pre-built wheels | ✅ via `huggingface-hub` pip |
| **API Key needed** | ❌ No | ⚠️ Optional (free HF token) |
| **Offline** | ✅ Fully offline | ❌ API call (but edge-tts fallback is always offline) |
| **Voice style** | Generic US English | British sports commentator |

---

## Environment Variables

| Variable | Service | Required | Description |
|----------|---------|----------|-------------|
| `DATABASE_URL` | Orchestrator | ✅ | PostgreSQL connection string |
| `GEMINI_API_KEY` | Inference | ✅ | Google AI Studio key for Gemini 2.0 Flash |
| `HF_TOKEN` | Both | ⚠️ Recommended | HuggingFace token for Kokoro-82M TTS |
| `ORCHESTRATOR_URL` | Inference | ❌ | Default: `http://localhost:4000` |
| `INFERENCE_URL` | Orchestrator | ❌ | Default: `http://localhost:8000` |
| `CORS_ORIGIN` | Orchestrator | ❌ | Default: `http://localhost:3000` |
| `PORT` | Orchestrator | ❌ | Default: `4000` |

---

## GPU Support

The inference service uses **CUDA 12.4** for GPU acceleration:

```python
import torch
print(torch.cuda.is_available())        # True if GPU detected
print(torch.cuda.get_device_name(0))    # e.g., "NVIDIA GeForce RTX 3050"
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
```

### Kokoro TTS not working
Set `HF_TOKEN` in `services/inference/.env`. Without it, the system uses anonymous HuggingFace API with strict rate limits. If Kokoro is unavailable, `edge-tts` is used automatically.

### Heatmap not appearing in Analytics tab
The heatmap is only generated if YOLO tracked players during analysis. Check inference logs for:
- `Heatmap saved →` (success)
- `Heatmap generation failed:` (error — re-analyze the match)

### Prisma `generate` fails with EPERM
The orchestrator process has the Prisma DLL locked. Stop the service first, then run `npx prisma generate`, then restart.

---

## File Structure

```
Matcha-AI-DTU/
├── apps/
│   └── web/               # Next.js frontend
├── services/
│   ├── orchestrator/      # NestJS API + WebSocket
│   │   └── prisma/        # Schema + migrations
│   └── inference/         # FastAPI + YOLO + Kokoro TTS
│       ├── app/core/      # analysis.py, heatmap.py, goal_detection.py
│       ├── venv/          # Python virtualenv (gitignored)
│       ├── yolov8s.pt     # YOLOv8 small model weights
│       └── uploads/       # Uploaded videos + generated assets
├── uploads/               # Shared upload directory
└── docker-compose.yml     # PostgreSQL + Redis
```
