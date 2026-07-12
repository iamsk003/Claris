import { useEffect, useState } from "react";
import { TopBar } from "../components/TopBar";
import { DropZone } from "../components/DropZone";
import { HistoryPanel } from "../components/HistoryPanel";
import { startRun, uploadClip } from "../api/client";
import { api } from "../config";
import { useRunStore } from "../store/useRunStore";
import { useNavigate, Link } from "../router";
import { bytes } from "../lib/format";
import { captureThumbnail, putPending } from "../lib/history";
import { DEMO_RUN_ID } from "../demo/sampleRun";

export function Upload() {
  const navigate = useNavigate();
  const setClip = useRunStore((s) => s.setClip);
  const setRun = useRunStore((s) => s.setRun);
  const setMode = useRunStore((s) => s.setMode);
  const reset = useRunStore((s) => s.reset);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"file" | "url">("file");
  const [url, setUrl] = useState("");
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function choose(f: File, from: string | null = null) {
    setError(null);
    setFile(f);
    setSourceUrl(from);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function loadUrl() {
    const u = url.trim();
    if (!/^https?:\/\/\S+/i.test(u)) {
      setError("Enter a direct http(s) link to a video file.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      // Go through the backend proxy: the browser cannot fetch most public video hosts
      // directly (CORS), but the server can download them and stream the bytes back.
      const res = await fetch(api.fetchVideo(u));
      if (!res.ok) {
        const msg = await res.json().then((d) => d?.error).catch(() => null);
        throw new Error(msg || `Could not load that URL (${res.status}).`);
      }
      const blob = await res.blob();
      const name = decodeURIComponent(u.split("/").pop()?.split("?")[0] || "video.mp4");
      choose(new File([blob], name, { type: blob.type || "video/mp4" }), u);
    } catch (e) {
      setError(
        (e instanceof Error ? e.message : "Could not load that URL") +
          " — use a direct, publicly accessible .mp4/.webm link (not YouTube, Google Drive, or a social share page).",
      );
    } finally {
      setBusy(false);
    }
  }

  async function generate() {
    if (!file || !previewUrl) return;
    setBusy(true);
    setError(null);
    reset();
    try {
      const { clip_id } = await uploadClip(file, setProgress);
      setClip(clip_id, previewUrl);
      const { run_id } = await startRun(clip_id);
      setRun(run_id);
      setMode("live");
      const thumbnail = await captureThumbnail(previewUrl);
      putPending(run_id, {
        label: sourceUrl ?? file.name,
        source: sourceUrl ? "url" : "file",
        thumbnail,
      });
      navigate(`/processing/${run_id}`);
    } catch (e) {
      setError(
        (e instanceof Error ? e.message : "Upload failed") +
          " — is the backend running at VITE_API_URL? You can still run the bundled sample.",
      );
      setBusy(false);
    }
  }

  return (
    <div className="min-h-full">
      <TopBar />
      <main className="mx-auto max-w-3xl px-6 py-12">
        <h1 className="text-2xl font-semibold text-bay-ink">Upload a clip</h1>
        <p className="mt-1 text-sm text-bay-ink-2">
          A 30-second to two-minute MP4 works best. Nothing is stored beyond this run.
        </p>

        <div className="mt-6">
          {!file ? (
            <div className="space-y-4">
              <div className="inline-flex rounded-lg border border-bay-line p-0.5 text-sm">
                {(["file", "url"] as const).map((t) => (
                  <button
                    key={t}
                    className={`rounded-md px-3 py-1.5 ${
                      tab === t ? "bg-signal/10 text-signal" : "text-bay-ink-2"
                    }`}
                    onClick={() => {
                      setTab(t);
                      setError(null);
                    }}
                  >
                    {t === "file" ? "Upload file" : "Paste video URL"}
                  </button>
                ))}
              </div>

              {tab === "file" ? (
                <DropZone onFile={(f) => choose(f)} />
              ) : (
                <div className="panel p-4">
                  <label className="text-xs text-bay-ink-2" htmlFor="video-url">
                    Direct link to a publicly accessible video (http/https)
                  </label>
                  <div className="mt-2 flex gap-2">
                    <input
                      id="video-url"
                      type="url"
                      inputMode="url"
                      placeholder="https://example.com/clip.mp4"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") loadUrl();
                      }}
                      className="min-w-0 flex-1 rounded-md border border-bay-line bg-transparent px-3 py-2 text-sm text-bay-ink outline-none focus:border-signal"
                      disabled={busy}
                    />
                    <button className="btn btn-signal px-4" onClick={loadUrl} disabled={busy || !url.trim()}>
                      {busy ? "Loading…" : "Load"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="panel overflow-hidden">
              <video src={previewUrl!} controls className="max-h-[46vh] w-full bg-black" />
              <div className="flex items-center justify-between gap-3 border-t border-bay-line px-4 py-3">
                <div className="min-w-0">
                  <div className="truncate text-sm text-bay-ink">{file.name}</div>
                  <div className="num text-xs text-bay-ink-3">{bytes(file.size)}</div>
                </div>
                <button
                  className="btn px-3 py-1.5 text-xs"
                  onClick={() => {
                    setFile(null);
                    setSourceUrl(null);
                    if (previewUrl) URL.revokeObjectURL(previewUrl);
                    setPreviewUrl(null);
                  }}
                  disabled={busy}
                >
                  Choose another
                </button>
              </div>
            </div>
          )}
        </div>

        {file && (
          <div className="mt-5">
            <button className="btn btn-signal w-full py-3 text-base" onClick={generate} disabled={busy}>
              {busy ? `Uploading… ${Math.round(progress * 100)}%` : "Generate captions"}
            </button>
            {busy && (
              <div className="mt-3 h-1 w-full overflow-hidden rounded bg-bay-line">
                <div
                  className="h-full bg-signal transition-[width]"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mt-5 rounded-lg border border-lane-audio/40 bg-lane-audio/5 p-4">
            <p className="text-sm text-bay-ink">{error}</p>
            <Link to={`/processing/${DEMO_RUN_ID}`} className="btn mt-3 inline-flex">
              Run the bundled sample
            </Link>
          </div>
        )}

        <HistoryPanel />
      </main>
    </div>
  );
}
