# Matcha-AI-DTU Frontend (`apps/web`)

This is the frontend application for the Matcha-AI-DTU platform, built with modern web technologies including **Next.js 15**, **React 19**, and **Tailwind CSS**. 

## ✨ Features
- **Video Upload Portal**: Utilizes `react-dropzone` for robust and intuitive drag-and-drop video file uploading.
- **Real-Time Dashboards**: Establishes continuous connection to the Orchestrator utilizing `socket.io-client` for real-time video processing updates and analysis percentages.
- **Client-Side AI Integration**: Features TensorFlow.js (`@tensorflow/tfjs`, `@tensorflow-models/coco-ssd`) and FFmpeg WASM interfaces for client-side preprocessing.
- **Sleek UI Framework**: Tailored with extensive Tailwind CSS styling and animated icons from `lucide-react`.

## 🛠 Tech Stack
- **Framework**: Next.js (App Router)
- **UI & Styling**: Tailwind CSS v4, `clsx`, `tailwind-merge`
- **Data/WebSockets**: Socket.io Client
- **Processing Support**: FFmpeg integration

## 🚀 Getting Started

From the root of the repository or from `apps/web`:

### Installation
Dependencies are gracefully handled by turbo in the root workspace, but you can install locally too:
```bash
npm install
```

### Running Locally
```bash
npm run dev
```

The application will launch on `http://localhost:3000`. 
Ensure the Orchestrator API (`http://localhost:4000`) is running concurrently for full functionality, as the frontend heavily relies on it for API requests and WebSocket events.

## 📁 Directory Context
- `app/`: Next.js App Router endpoints, pages, and layouts (e.g. `page.tsx`, `globals.css`).
- Components include modern Hooks-based React designs managing websocket states, highlighting parsed game events, and rendering generated commentary audio on top of analyzed sports videos.
