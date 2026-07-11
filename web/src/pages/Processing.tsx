import { useEffect, useRef, useState } from "react";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { AnimatePresence, motion } from "framer-motion";
import { TopBar } from "../components/TopBar";
import { api } from "../config";
import { useRunStore } from "../store/useRunStore";
import { useNavigate } from "../router";
import { useDemoStream } from "../demo/useDemoStream";
import { DEMO_RUN_ID, isDemoRunId } from "../demo/sampleRun";
import type { RunEvent } from "../types";

// Presentation-only status copy. The real progress/completion signal is still the backend
// event stream (handled by the unchanged effects below); these phrases are a smooth overlay.
const PHRASES = [
  "Analyzing video…",
  "Extracting frames…",
  "Reading on-screen text…",
  "Understanding visuals…",
  "Listening to speech…",
  "Generating captions…",
  "Preparing results…",
];

export function Processing({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const applyEvent = useRunStore((s) => s.applyEvent);
  const reset = useRunStore((s) => s.reset);
  const setMode = useRunStore((s) => s.setMode);
  const stages = useRunStore((s) => s.stages);
  const events = useRunStore((s) => s.events);

  const startsDemo = isDemoRunId(runId);
  const [demo, setDemo] = useState(startsDemo);
  const gotEvent = useRef(false);
  const navigated = useRef(false);

  // Fresh timeline for this run.
  useEffect(() => {
    reset();
    setMode(startsDemo ? "demo" : "live");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  function goResults(target: string) {
    if (navigated.current) return;
    navigated.current = true;
    window.setTimeout(() => navigate(`/results/${target}`), 650);
  }

  // Live WebSocket. Never connects for a demo run.
  const { lastJsonMessage, readyState } = useWebSocket(
    api.events(runId),
    { shouldReconnect: () => false, retryOnError: false, share: false },
    !demo,
  );

  useEffect(() => {
    if (!lastJsonMessage) return;
    gotEvent.current = true;
    const e = lastJsonMessage as RunEvent;
    applyEvent(e);
    if (e.stage === "done") goResults(runId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastJsonMessage]);

  // If the socket can't be reached and no events arrive, fall back to labeled demo mode.
  useEffect(() => {
    if (demo) return;
    const id = window.setTimeout(() => {
      if (!gotEvent.current && readyState !== ReadyState.OPEN) {
        setDemo(true);
        setMode("demo");
      }
    }, 3500);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, readyState]);

  // Demo replay. On done, always route to the bundled sample result.
  useDemoStream(demo, applyEvent, () => goResults(DEMO_RUN_ID));

  // ---- presentation only: smooth status-text progression ----------------- #
  const [phase, setPhase] = useState(0);

  // March through the phrases so the copy always feels alive; hold at "Generating captions…"
  // until the backend signals the run is finishing.
  useEffect(() => {
    const id = window.setInterval(() => {
      if (!navigated.current) setPhase((p) => (p < 5 ? p + 1 : p));
    }, 1700);
    return () => clearInterval(id);
  }, []);

  // A finishing event jumps straight to "Preparing results…".
  useEffect(() => {
    const last = events[events.length - 1];
    if (last && /(finish|done|prepar|select)/.test((last.stage || "").toLowerCase())) {
      setPhase(6);
    }
  }, [events]);

  const connecting = !demo && readyState === ReadyState.CONNECTING && !gotEvent.current;
  const complete = stages.done?.status === "done";
  const progress = complete ? 1 : (phase + 1) / PHRASES.length;

  return (
    <div className="min-h-full">
      <TopBar />
      <main className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-2xl flex-col items-center justify-center px-6 py-8">
        <ProgressRing progress={progress} />

        <div className="mt-6 h-8 text-center">
          <AnimatePresence mode="wait">
            <motion.p
              key={phase}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
              className="text-lg font-medium tracking-tight text-bay-ink"
            >
              {PHRASES[phase]}
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Phase dots */}
        <div className="mt-5 flex items-center gap-2">
          {PHRASES.map((_, i) => (
            <span
              key={i}
              className="h-1.5 rounded-full transition-all duration-500"
              style={{
                width: i === phase ? 22 : 6,
                backgroundColor:
                  i < phase || complete
                    ? "rgba(255,176,58,0.8)"
                    : i === phase
                      ? "#ffb03a"
                      : "rgba(255,255,255,0.14)",
              }}
            />
          ))}
        </div>

        <p className="num mt-6 text-[11px] uppercase tracking-widest text-bay-ink-3">
          {demo
            ? "sample data · replaying a recorded run"
            : connecting
              ? "connecting to backend…"
              : "grounding every caption in the evidence"}
        </p>
      </main>
    </div>
  );
}

/** Animated CLARIS mark inside a circular progress ring. */
function ProgressRing({ progress }: { progress: number }) {
  const R = 82;
  const C = 2 * Math.PI * R;
  return (
    <div className="relative h-44 w-44 sm:h-52 sm:w-52">
      {/* soft pulsing halo */}
      <div className="absolute inset-6 rounded-full bg-signal/10 blur-2xl animate-pulse-signal" />

      <svg viewBox="0 0 200 200" className="absolute inset-0 h-full w-full -rotate-90">
        <circle cx="100" cy="100" r={R} fill="none" stroke="#292d33" strokeWidth="3" />
        {/* determinate arc — advances with the run */}
        <circle
          cx="100"
          cy="100"
          r={R}
          fill="none"
          stroke="#ffb03a"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - Math.min(1, Math.max(0, progress)))}
          style={{ transition: "stroke-dashoffset 0.8s ease", filter: "drop-shadow(0 0 6px rgba(255,176,58,0.5))" }}
        />
      </svg>

      {/* indeterminate spinner arc for continuous motion */}
      <svg viewBox="0 0 200 200" className="absolute inset-0 h-full w-full animate-spin" style={{ animationDuration: "2.6s" }}>
        <circle
          cx="100"
          cy="100"
          r={R - 9}
          fill="none"
          stroke="#ffb03a"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * (R - 9) * 0.16} ${2 * Math.PI * (R - 9)}`}
          opacity="0.55"
        />
      </svg>

      {/* animated logo mark */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <svg
            width="70"
            height="70"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            style={{ filter: "drop-shadow(0 0 10px rgba(255,176,58,0.35))" }}
          >
            <rect x="2.5" y="4.5" width="19" height="15" rx="2.5" stroke="#e7e9ec" strokeWidth="1.2" />
            <motion.line
              x1="14"
              y1="3.5"
              x2="14"
              y2="20.5"
              stroke="#ffb03a"
              strokeWidth="1.6"
              animate={{ x1: [9, 18, 9], x2: [9, 18, 9] }}
              transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
            />
            <circle cx="8" cy="12" r="2.4" stroke="#e7e9ec" strokeWidth="1.2" />
          </svg>
        </motion.div>
      </div>
    </div>
  );
}
