// Picks a video file using expo-document-picker and uploads it to the orchestrator
// Returns upload progress and the resulting matchId
// TODO: implement with expo-document-picker + XMLHttpRequest for progress tracking
export function useVideoUpload() {
  return {
    pickAndUpload: async () => {},
    uploadProgress: 0,
    status: "idle" as "idle" | "uploading" | "processing" | "success" | "error",
    matchId: null as string | null,
  };
}
