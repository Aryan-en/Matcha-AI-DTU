// Thin HTTP + WebSocket client wrapping the Matcha AI orchestrator REST API
// All fetch calls go through here so the base URL is in one place

import { API_BASE } from "@/constants/api";

export const api = {
  getMatches: () => fetch(`${API_BASE}/matches`).then(r => r.json()),
  getMatch: (id: string) => fetch(`${API_BASE}/matches/${id}`).then(r => r.json()),
  deleteMatch: (id: string) => fetch(`${API_BASE}/matches/${id}`, { method: "DELETE" }),
  reanalyze: (id: string) => fetch(`${API_BASE}/matches/${id}/reanalyze`, { method: "POST" }),
};
