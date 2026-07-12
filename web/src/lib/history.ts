// Local, browser-only history of processed clips. Nothing here touches the backend: the
// full result envelope is cached in localStorage so a run can be reopened even after the
// server has evicted its clip. Scoped to this browser only — no login, no sync.

import type { RunResult, StyledCaption } from "../types";

const HISTORY_KEY = "claris.history.v1";
const PENDING_KEY = "claris.history.pending.v1";
const MAX_ENTRIES = 24;

export interface HistoryEntry {
  runId: string;
  label: string; // filename or URL
  source: "file" | "url";
  time: number; // ms epoch, when processing completed
  thumbnail?: string; // data URL, best-effort
  captions: StyledCaption[];
  result: RunResult; // full envelope, for offline reopen
}

type PendingMeta = { label: string; source: "file" | "url"; thumbnail?: string };

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota or disabled storage — history is best-effort */
  }
}

export function loadHistory(): HistoryEntry[] {
  return read<HistoryEntry[]>(HISTORY_KEY, []);
}

export function removeEntry(runId: string): HistoryEntry[] {
  const next = loadHistory().filter((e) => e.runId !== runId);
  write(HISTORY_KEY, next);
  return next;
}

export function clearHistory(): void {
  write(HISTORY_KEY, []);
}

export function getHistoryResult(runId: string): RunResult | undefined {
  return loadHistory().find((e) => e.runId === runId)?.result;
}

/** Stash upload-time metadata (label/source/thumbnail) until the run's result arrives. */
export function putPending(runId: string, meta: PendingMeta): void {
  const map = read<Record<string, PendingMeta>>(PENDING_KEY, {});
  map[runId] = meta;
  write(PENDING_KEY, map);
}

function takePending(runId: string): PendingMeta | undefined {
  const map = read<Record<string, PendingMeta>>(PENDING_KEY, {});
  const meta = map[runId];
  if (meta) {
    delete map[runId];
    write(PENDING_KEY, map);
  }
  return meta;
}

/** Commit a completed run to history (idempotent; keeps the first-seen entry). */
export function recordFromResult(runId: string, result: RunResult): void {
  const list = loadHistory();
  if (list.some((e) => e.runId === runId)) return;
  const pending = takePending(runId);
  const source = pending?.source ?? "file";
  const label = pending?.label ?? result.task_id;
  // A remote URL survives reloads, so replay it directly; uploaded blobs cannot.
  const stored: RunResult =
    source === "url" ? { ...result, video_url: label } : result;
  const entry: HistoryEntry = {
    runId,
    label,
    source,
    time: Date.now(),
    thumbnail: pending?.thumbnail,
    captions: result.captions,
    result: stored,
  };
  write(HISTORY_KEY, [entry, ...list].slice(0, MAX_ENTRIES));
}

/** Best-effort poster frame from a playable video URL. Undefined on CORS taint or error. */
export async function captureThumbnail(src: string): Promise<string | undefined> {
  return new Promise((resolve) => {
    const done = (v?: string) => resolve(v);
    try {
      const video = document.createElement("video");
      video.crossOrigin = "anonymous";
      video.muted = true;
      video.preload = "metadata";
      video.src = src;
      const timer = window.setTimeout(() => done(undefined), 3000);
      video.onloadeddata = () => {
        try {
          video.currentTime = Math.min(1, (video.duration || 2) / 2);
        } catch {
          window.clearTimeout(timer);
          done(undefined);
        }
      };
      video.onseeked = () => {
        window.clearTimeout(timer);
        try {
          const w = 160;
          const ratio = video.videoHeight / (video.videoWidth || 1) || 0.5625;
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = Math.round(w * ratio);
          const ctx = canvas.getContext("2d");
          if (!ctx) return done(undefined);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          done(canvas.toDataURL("image/jpeg", 0.6));
        } catch {
          done(undefined);
        }
      };
      video.onerror = () => {
        window.clearTimeout(timer);
        done(undefined);
      };
    } catch {
      done(undefined);
    }
  });
}
