// The single seam for backend integration. Every fetch URL and the WebSocket URL derive
// from API_URL; nothing else in the app hardcodes a host.

const raw = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export const API_URL = raw.replace(/\/+$/, "");

/** ws:// or wss:// origin derived from the HTTP base, for the event stream. */
export const WS_URL = API_URL.replace(/^http/, "ws");

export const api = {
  clips: () => `${API_URL}/api/clips`,
  run: (clipId: string) => `${API_URL}/api/clips/${clipId}/run`,
  events: (runId: string) => `${WS_URL}/api/runs/${runId}/events`,
  result: (runId: string) => `${API_URL}/api/runs/${runId}`,
  clipVideo: (clipId: string) => `${API_URL}/api/clips/${clipId}/video`,
  fetchVideo: (u: string) => `${API_URL}/api/fetch-video?url=${encodeURIComponent(u)}`,
};

/** Soft cap: warn above this, never block. */
export const MAX_CLIP_BYTES = 200 * 1024 * 1024;
