// Base URL for the Matcha AI orchestrator.
// Switch to your deployed URL for production builds.
export const API_BASE = "http://10.0.2.2:4000"; // Android emulator → localhost
export const WS_URL = "http://10.0.2.2:4000";

// Status display config — mirrors web STATUS_CONFIG
export const STATUS_COLORS = {
  COMPLETED: "#34d399",
  PROCESSING: "#60a5fa",
  UPLOADED: "#fbbf24",
  FAILED: "#f87171",
} as const;
