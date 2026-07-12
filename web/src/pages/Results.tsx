import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { TopBar } from "../components/TopBar";
import { VideoPlayer } from "../components/VideoPlayer";
import { EvidenceTimeline } from "../components/EvidenceTimeline";
import { CaptionCard } from "../components/CaptionCard";
import { useRunResult } from "../api/hooks";
import { useRunStore } from "../store/useRunStore";
import { useInspectStore } from "../store/useInspectStore";
import { indexById } from "../lib/evidence";
import { recordFromResult } from "../lib/history";
import { api } from "../config";
import { isDemoRunId } from "../demo/sampleRun";
import { ALL_STYLES, type StyledCaption } from "../types";
import { pct } from "../lib/format";
import { Link } from "../router";

export function Results({ runId }: { runId: string }) {
  const { data, isLoading, isError, error } = useRunResult(runId);
  const localVideoUrl = useRunStore((s) => s.localVideoUrl);
  const clearPin = useInspectStore((s) => s.clearPin);
  const clearCaption = useInspectStore((s) => s.clearCaption);
  const requestSeek = useInspectStore((s) => s.requestSeek);

  const [currentTime, setCurrentTime] = useState(0);
  const isDemo = isDemoRunId(runId);

  // Reset cross-highlight when leaving; ESC clears any pin.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        clearPin();
        clearCaption();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      clearPin();
      clearCaption();
    };
  }, [clearPin, clearCaption]);

  // Persist completed runs to browser-local history (skip the bundled sample).
  useEffect(() => {
    if (data && !isDemo) recordFromResult(runId, data);
  }, [data, isDemo, runId]);

  const byId = useMemo(() => (data ? indexById(data.ledger.items) : new Map()), [data]);

  if (isLoading) return <Shell><LoadingState /></Shell>;
  if (isError || !data)
    return (
      <Shell>
        <EmptyState
          title="Couldn't load this run"
          body={(error as Error)?.message ?? "The result isn't available."}
        />
      </Shell>
    );

  const duration = data.ledger.video_meta.duration_s || 1;
  const captions = orderCaptions(data.captions);
  const videoSrc =
    data.video_url ??
    (data.clip_id && !isDemo ? api.clipVideo(data.clip_id) : undefined) ??
    (!isDemo ? localVideoUrl : undefined);

  const flags = data.ledger.modality_flags;
  const activeModalities = [
    flags.has_speech && "speech",
    flags.has_visual && "visual",
    flags.has_ocr && "ocr",
    flags.has_audio_event && "audio",
    flags.has_motion && "motion",
  ].filter(Boolean) as string[];

  return (
    <Shell>
      {/* Sub-header */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold text-bay-ink">Results</h1>
          <span className="num text-xs text-bay-ink-3">{data.task_id}</span>
          {isDemo && (
            <span className="chip border-signal/50 text-signal">
              <span className="h-1.5 w-1.5 rounded-full bg-signal" /> sample data
            </span>
          )}
          {data.degraded && (
            <span className="chip border-lane-audio/50 text-lane-audio">degraded run</span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="chip">coverage {pct(data.ledger.coverage)}</span>
          {activeModalities.map((m) => (
            <span key={m} className="chip">
              {m}
            </span>
          ))}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.02fr)]">
        {/* Captions — left on desktop, below on mobile */}
        <section className="order-2 space-y-4 lg:order-1">
          {captions.map((c, i) => (
            <motion.div
              key={c.style}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06, duration: 0.4, ease: "easeOut" }}
            >
              <CaptionCard
                caption={c}
                candidates={data.candidates?.[c.style] ?? []}
                byId={byId}
                index={i}
              />
            </motion.div>
          ))}

          <p className="num px-1 text-[11px] leading-relaxed text-bay-ink-3">
            Hover a sentence to light the evidence that supports it and seek the video to its
            first source. Click any evidence block to pin it — every caption that cites it
            highlights. Press Esc to clear.
          </p>
        </section>

        {/* Video + evidence timeline — right on desktop, top on mobile */}
        <section className="order-1 space-y-4 lg:order-2">
          <div className="lg:sticky lg:top-20 lg:space-y-4">
            <VideoPlayer
              src={videoSrc}
              duration={duration}
              currentTime={currentTime}
              onTime={setCurrentTime}
            />
            <EvidenceTimeline
              items={data.ledger.items}
              duration={duration}
              currentTime={currentTime}
              onSeek={(t) => {
                setCurrentTime(t);
                requestSeek(t);
              }}
            />
            <div className="num flex flex-wrap gap-x-3 gap-y-1 px-1 text-[11px] text-bay-ink-3">
              <span>models:</span>
              {data.ledger.perception_models.map((m) => (
                <span key={m} className="text-bay-ink-2">
                  {m}
                </span>
              ))}
            </div>
          </div>
        </section>
      </div>
    </Shell>
  );
}

function orderCaptions(caps: StyledCaption[]): StyledCaption[] {
  const byStyle = new Map(caps.map((c) => [c.style, c]));
  return ALL_STYLES.map((s) => byStyle.get(s)).filter(Boolean) as StyledCaption[];
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <TopBar />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">{children}</main>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="space-y-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="panel h-40 animate-pulse-signal opacity-40" />
        ))}
      </div>
      <div className="panel h-96 animate-pulse-signal opacity-40" />
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="mx-auto max-w-lg py-24 text-center">
      <div className="num text-xs uppercase tracking-widest text-bay-ink-3">no result</div>
      <h1 className="mt-3 text-2xl font-semibold text-bay-ink">{title}</h1>
      <p className="mt-2 text-bay-ink-2">{body}</p>
      <div className="mt-6 flex justify-center gap-3">
        <Link to="/upload" className="btn">
          Upload a clip
        </Link>
        <Link to="/results/demo" className="btn btn-signal">
          View the sample
        </Link>
      </div>
    </div>
  );
}
