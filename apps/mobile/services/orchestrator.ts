// Thin HTTP + WebSocket client wrapping the Matcha AI orchestrator REST API
// All fetch calls go through here so the base URL is in one place

import { API_BASE } from "@/constants/api";
import { createApiClient } from "@matcha/shared";

export const api = createApiClient(API_BASE);

// Mobile-specific upload helper (React Native requires different FormData handling)
export const uploadVideoMobile = async (uri: string, name: string, type: string) => {
  const formData = new FormData();
  // @ts-ignore - React Native FormData expects an object for files
  formData.append("file", { uri, name, type });

  const res = await fetch(`${API_BASE}/matches/upload`, {
    method: "POST",
    body: formData,
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  if (!res.ok) throw new Error("Upload failed");
  return res.json();
};
