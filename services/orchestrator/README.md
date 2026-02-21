# Matcha-AI-DTU Orchestrator (`services/orchestrator`)

The Orchestrator acts as the central hub of the Matcha-AI-DTU monorepo. Built aggressively leveraging the **NestJS** framework for its modularity and robust API routing, this service primarily manages communication between the web frontend and the Python AI Inference engine.

## 🔗 The Central Link
- **Coordinates the Flow**: The primary gateway `http://localhost:4000`. Receives uploaded sports footage, delegates video analysis to the Inference Engine, and then pushes structured progress updates directly to the frontend.
- **WebSocket Gateway**: Integrates extensively with `socket.io`. When Inference reports back processed statuses or identified events, the Orchestrator instantly emits these updates to Next.js clients avoiding active polling.
- **Data Persistence**: Configured seamlessly with **PostgreSQL** mapping relational tables over **Prisma ORM**. Connects to a cache/pub-sub system running **Redis**.

## 🛠 Tech Stack
- **Backend Framework**: NestJS (`@nestjs/core`, `@nestjs/common`)
- **Real-time Server**: Socket.io (`@nestjs/websockets`, `@nestjs/platform-socket.io`)
- **Database ORM**: Prisma Client (`@prisma/client`)
- **HTTP Client/Requests**: Axios (`@nestjs/axios`)
- **Cache/Background**: Redis

## 🚀 Setup & Execution

### Running the Environment
You must have the Docker containers (PostgreSQL + Redis) spawned beforehand:
```bash
# In the project root directory
docker compose up -d
```

### Database Initialization
Apply Prisma models over the running Postgres container:
```bash
npx prisma generate
npx prisma migrate deploy
```

### Service Launch

Install dependencies if not already run at the workspace root:
```bash
npm install
```

Run in development/watch mode:
```bash
npm run start:dev
```
```bash
# Debug Mode
npm run start:debug
```

The server binds natively to port **`4000`**. You can hit it from `apps/web` components directly.

## 📝 Service Design

The `services/orchestrator` operates via distinct Controllers, Services, and Gateways.
It essentially wraps around `app.module.ts` separating logic based on:
1. Video Intake Routes (`HTTP POST`) 
2. Real-time Status (`WebSocket events`) 
3. Inference Return Hooks (`HTTP Callbacks`) 
4. Media Hosting (`Uploads` Static Directory Delivery)
