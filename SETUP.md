# Matcha AI: Comprehensive Setup Guide

This guide provides a detailed, step-by-step walkthrough for setting up the Matcha AI monorepo on a fresh local machine.

## 📋 Prerequisites

Matcha requires the following system-level tools to be installed and available in your `PATH`:

| Tool | Version | Purpose |
| :--- | :--- | :--- |
| **Node.js** | 18+ | Runtime for Frontend, Orchestrator, and Shared tooling. |
| **Python** | 3.11+ | Runtime for the Inference service (PyTorch/YOLO/Gemini). |
| **Docker** | Latest | Container engine for the PostgreSQL and Redis data layer. |
| **FFmpeg** | Latest | Essential for video processing, highlight cutting, and TTS muxing. |
| **NVIDIA GPU** | (Optional) | Recommended for real-time video analysis via CUDA 12.4. |

---

## 🏗️ 1. Infrastructure (Docker)

Matcha uses Docker Compose to manage its persistent data and caching layers.

1.  Navigate to the project root: `cd Matcha-AI-DTU`
2.  Start the infrastructure: `docker compose up -d`

| Service | Port | Internal Details |
| :--- | :--- | :--- |
| **PostgreSQL** | `5433` | Primary database for matches, users, and events. |
| **Redis** | `6380` | High-performance cache for active analysis tracking. |

---

## 📦 2. JavaScript Monorepo Setup

The platform uses **Turborepo** to manage dependencies and build pipelines across its apps and 6 shared packages.

1.  **Install Dependencies**: Install all packages and link the monorepo workspace.
    ```bash
    npm install
    ```

2.  **Initialize Shared Database**: Set up the centralized `@matcha/database` package.
    ```bash
    # Generate the shared Prisma client
    npx turbo run generate

    # Deploy the initial database schema
    npx turbo run db:migrate
    ```

3.  **Build Shared Packages**: Ensure all packages (`ui`, `theme`, `shared`, etc.) are compiled.
    ```bash
    npx turbo run build
    ```

---

## 🧠 3. Python Inference Setup

The inference service houses the AI pipeline and requires a isolated Python environment.

1.  Navigate to the service: `cd services/inference`
2.  Create a virtual environment: `py -3.11 -m venv venv`
3.  Activate the environment:
    - **Windows**: `.\venv\Scripts\activate`
    - **Unix/Mac**: `source venv/bin/activate`
4.  Install dependencies:
    ```bash
    # Upgrade pip first
    pip install --upgrade pip

    # Install PyTorch with CUDA 12.4 (if GPU available)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Install remaining requirements
    pip install -r requirements.txt
    ```

---

## 🔐 4. Configuration (Environment Variables)

Matcha uses strict, boot-time environment validation. You MUST create the following `.env` files.

### Orchestrator (`services/orchestrator/.env`)
```env
# Database
DATABASE_URL="postgresql://matcha_user:matcha_password@localhost:5433/matcha_db?schema=public"

# AI Integration
GEMINI_API_KEY="your_google_ai_studio_api_key"
HF_TOKEN="hf_your_huggingface_token" # (Optional: For higher TTS rate limits)

# Networking
PORT=4000
CORS_ORIGIN="http://localhost:3000"
INFERENCE_URL="http://localhost:8000"
```

### Web App (`apps/web/.env.local`)
```env
NEXT_PUBLIC_API_URL="http://localhost:4000"
```

### Inference Service (`services/inference/.env`)
```env
ORCHESTRATOR_URL="http://localhost:4000"
GEMINI_API_KEY="your_google_ai_studio_api_key"
HF_TOKEN="hf_your_huggingface_token"
```

---

## 🚀 5. Running the Platform

### Option A: The "One Command" Developer Mode (Recommended)
From the root of the monorepo, run:
```bash
npx turbo run dev
```
This launches the **Web App (3000)**, **Orchestrator (4000)**, and **Inference (8000)** simultaneously, complete with hot-reloading and unified streaming logs.

### Option B: Manual Startup
If you need to debug a specific service, run them in separate terminals:
1.  **Frontend**: `cd apps/web && npm run dev`
2.  **Backend**: `cd services/orchestrator && npm run start:dev`
3.  **AI Engine**: `cd services/inference && uvicorn main:app --port 8000`

---

## ❓ Troubleshooting

### 1. Cross-Package Type Resolution
If your IDE (VS Code) shows "Module not found" for `@matcha/*` packages:
- **Solution**: Restart the **TypeScript Server** (Cmd+Shift+P → "Restart TS Server").
- **Cause**: Turborepo uses standard Node/monorepo resolution that IDEs иногда block.

### 2. Prisma Client Generation
If you see "PrismaClient did not find a schema":
- **Solution**: Run `npx turbo run generate` from the root. This ensures the client is generated into `@matcha/database/node_modules`.

### 3. Analysis Stuck at 0%
- **Check**: Is the Inference service log showing `Connection Refused`?
- **Solution**: Ensure `ORCHESTRATOR_URL` in `inference/.env` matches the port of your running NestJS service.

### 4. GPU Not Detected
- **Check**: Run `python -c "import torch; print(torch.cuda.is_available())"` in the inference venv.
- **Solution**: Ensure your NVIDIA Drivers are updated and you installed the correct PyTorch version with the `--index-url` specified in Step 3.

---

> [!TIP]
> **Pro Tip**: Use the **Matcha Analytics Tab** to verify that YOLO and Heatmap generation are working correctly. If the heatmap is empty, ensure the video contains clear player detection frames.

