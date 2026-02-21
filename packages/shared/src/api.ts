/** HTTP API client for the Matcha AI orchestrator.
 *  Import this in apps/web and apps/mobile — pass the base URL from your env/constants.
 */

import type { MatchSummary, MatchDetail } from "./types";

export function createApiClient(baseUrl: string) {
  return {
    getMatches: (): Promise<MatchSummary[]> =>
      fetch(`${baseUrl}/matches`).then((r) => r.json()),

    getMatch: (id: string): Promise<MatchDetail> =>
      fetch(`${baseUrl}/matches/${id}`).then((r) => r.json()),

    deleteMatch: (id: string): Promise<Response> =>
      fetch(`${baseUrl}/matches/${id}`, { method: "DELETE" }),

    reanalyze: (id: string): Promise<Response> =>
      fetch(`${baseUrl}/matches/${id}/reanalyze`, { method: "POST" }),

    uploadVideo: (file: File | Blob, onProgress?: (pct: number) => void): Promise<MatchSummary> =>
      new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const form = new FormData();
        form.append("video", file);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
          else reject(new Error(`Upload failed: ${xhr.status}`));
        };
        xhr.onerror = () => reject(new Error("Network error"));
        xhr.open("POST", `${baseUrl}/upload`);
        xhr.send(form);
      }),
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
