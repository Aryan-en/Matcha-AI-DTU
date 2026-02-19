---
description: how to run the full matcha-ai stack
---

# Running Matcha AI

Follow these steps to launch the entire ecosystem (Infrastructure, Backend, AI Inference, and Frontend).

## 1. Start Infrastructure & AI Service (Docker)
This starts Postgres, Redis, MinIO, and the Python Inference service in a stable 3.11 environment.

// turbo
```powershell
docker-compose up -d --build
```

## 2. Start Application Services (Turbo)
Run this from the root directory to start both the Orchestrator (NestJS) and the Web Frontend (Next.js) concurrently.

// turbo
```powershell
npx turbo run dev
```

## Service Access Points
- **Web Frontend**: [http://localhost:3000](http://localhost:3000) (or 3001 if 3000 is occupied)
- **Orchestrator API**: [http://localhost:4000](http://localhost:4000)
- **Inference API**: [http://localhost:8000](http://localhost:8000)
- **MinIO Console**: [http://localhost:9001](http://localhost:9001)

## Troubleshooting
- If Prisma types are missing, run: `cd services/orchestrator; npx prisma generate`
- Ensure the `uploads/` directory exists in the root for video processing.
