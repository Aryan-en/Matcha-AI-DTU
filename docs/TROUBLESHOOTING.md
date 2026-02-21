# 🛠 Troubleshooting Matcha-AI-DTU

If you run into issues launching the application or processing a video, refer to the most common resolutions below.

---

## 🛑 Docker Infrastructure Issues

### Problem: `docker compose up -d` fails or ports are bound
**Symptoms**: Docker cannot bind to `5433` (Postgres) or `6380` (Redis).
**Cause**: Another application or local database service is occupying the port.
**Resolution**: 
1. Run `netstat -ano | findstr :5433` (Windows) to identify the PID.
2. Terminate the blocking process via Task Manager or `taskkill /PID <id> /F`.
3. Alternatively, explicitly define different port mappings in the `docker-compose.yml` and mirror them in `services/orchestrator/.env`.

---

## 🐍 Python / Inference Service Issues

### Problem: Torch/Cuda uses excessive VRAM
**Symptoms**: A video fails midway through analysis with `RuntimeError: CUDA out of memory`.
**Cause**: High-resolution video frames (e.g., 4K) are overflowing GPU VRAM limits.
**Resolution**:
1. Open `services/inference/app/core/analysis.py`.
2. Locate the frame reading logic and ensure `cv2.resize()` scales frames down to a maximum of 720p or 1080p before passing them into the YOLO pipeline. 

### Problem: `piper-tts` fails to install on Windows
**Symptoms**: `pip install piper-tts` fails with obscure C++ compiler errors.
**Cause**: The local Python environment might be missing specific `.whl` builds or `onnxruntime` links.
**Resolution**:
Ensure you are running exactly **Python 3.11** or **Python 3.12**. `piper-tts` maintains pre-built binaries for these specific versions. A 3.10 or 3.13 env will mandate compiling from source.

### Problem: Progress sits at 0% and does not update
**Symptoms**: Video uploads successfully via Next.js, but the UI is stuck at "Beginning Analysis (0%)".
**Cause**: The Inference service isn't reachable, or the Orchestrator WebSocket URL is incorrect inside Python.
**Resolution**:
1. Verify `services/inference` is actively running on Port `8000` via Uvicorn.
2. In your Inference terminal, check the value of `$env:ORCHESTRATOR_URL`. It **must** exactly match the Orchestrator's address including the `http://` prefix. (e.g., `http://localhost:4000`).

---

## 🧑‍💻 Node.js & Prisma Issues

### Problem: Prisma Client reports missing tables
**Symptoms**: `PrismaClientKnownRequestError: Table 'Match' does not exist in the current database.`
**Cause**: Migrations were not applied to your active Postgres Docker container.
**Resolution**:
```bash
cd services/orchestrator
npx prisma generate
npx prisma migrate deploy
```

### Problem: `npm install` throws ERESOLVE conflicts
**Symptoms**: Older packages conflict with Next.js 15 or React 19.
**Cause**: Strict peer dependency checking in npm v10+.
**Resolution**:
We heavily rely on latest features. Ensure you run:
```bash
npm install --legacy-peer-deps
```
If errors persist inside `apps/web`.

---

Still stuck? Please check our `.github/ISSUE_TEMPLATE` and file a detailed bug report!
