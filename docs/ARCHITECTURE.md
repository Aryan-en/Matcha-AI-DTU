# 🏗 System Architecture

Matcha-AI-DTU utilizes a microservice-like **Monorepo** architecture leveraging specific languages for their native strengths: 
- **TypeScript/React** for dynamic, real-time UI mapping.
- **TypeScript/NestJS** for strictly-typed API gateways and WebSockets.
- **Python/FastAPI** for deep learning AI inference processing.

This document serves to visually explain how data flows across the monorepo when a user initiates a request.

---

## 🌊 The Core Video Analysis Data flow 

The primary end-to-end operation is ingesting a raw video, processing it through YOLO Computer Vision AI models and Large Language Models, and returning a generated Sports Highlight audio synthetic file dynamically to the browser. 

```mermaid
sequenceDiagram
    autonumber
    
    actor User as Client (Next.js App)
    participant Orch as Orchestrator (NestJS)
    participant Redis as Redis Cache
    participant DB as PostgreSQL (Prisma)
    participant ML as Inference Engine (Python)

    %% -------------------------------------
    %% Stage 1: Upload & Registration Focus
    %% -------------------------------------
    User->>Orch: POST /matches (Uploads Video File)
    activate Orch
    
    Orch->>DB: Create Match Record & Return ID
    Orch->>Redis: Set Processing State [Analysis Queue]
    
    Orch->>ML: POST /analyze { videoPath, matchId }
    activate ML
    
    Orch-->>User: HTTP 201 Created { matchId }
    deactivate Orch
    
    %% -------------------------------------
    %% Stage 2: Processing & Live Updates
    %% -------------------------------------
    
    note over ML,User: Real-Time Event Loop via WebSockets
    
    loop Per Video Frame Batch
        ML->>ML: Run Phase 1b: Goal Detection Pipeline
        ML->>ML: Run Phase 2: Action parsing 
        
        ML->>Orch: POST /callbacks/progress { matchId, progress: x% }
        Orch->>User: WS Event "analysisProgress" [Progress Bar Update]
        
        opt Event Detected (e.g. GOAL!)
            ML->>Orch: POST /callbacks/event { ...EventData }
            Orch->>DB: Save Event Record
            Orch->>User: WS Event "eventDetected" [Live UI Popup]
        end
    end
    
    %% -------------------------------------
    %% Stage 3: Generative AI & Audio Render
    %% -------------------------------------
    
    ML->>ML: Formulate script using Google Gemini
    ML->>ML: Render voiceover via Piper TTS (.wav)
    
    ML->>Orch: POST /callbacks/complete { highlightMetadata }
    deactivate ML
    activate Orch
    
    Orch->>DB: Update Match Status = "COMPLETED"
    Orch->>Redis: Clear Cache Queue
    
    Orch-->>User: WS Event "analysisComplete" [Refresh App Data]
    deactivate Orch
    
    User->>Orch: GET /matches/{id}
    Orch-->>User: Return full payload with AI Audio URL
```

---

## 🗄️ Database Schema Diagram

We utilize Prisma ORM for type-safe database queries. The central entities revolve around `Match` (a video entity) and its children `MatchEvent`.

```mermaid
erDiagram
    MATCH {
        String id PK
        String title
        String videoUrl
        String ttsAudioUrl
        String status
        DateTime createdAt
        DateTime updatedAt
    }

    MATCH_EVENT {
        String id PK
        String matchId FK
        String eventType "GOAL, TACKLE, FOUL, etc."
        Float timestampSeconds
        Float confidence
        String description
    }
    
    MATCH ||--o{ MATCH_EVENT : "has many"
```

---

## 🔌 WebSockets Implementation (Socket.io)

For real-time progression we decouple the heavy Python operations from holding HTTP connections open using Callbacks and WebSockets.

1. **NestJS** mounts a Socket.IO Gateway on port `4000`.
2. When the **Inference (Python)** script processes frames, it issues a synchronous *fire-and-forget* HTTP request to the Orchestrator (`/callbacks/progress`).
3. The **Orchestrator** translates this HTTP payload into an active Socket Event mapped to all connected Next.js users listening to that namespace.
4. If a WebSocket disconnects, the analysis **continues uninterrupted** inside the Python environment, preventing dropped progress upon a user refreshing their browser.
