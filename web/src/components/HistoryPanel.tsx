import { useState } from "react";
import { Link } from "../router";
import { clearHistory, loadHistory, removeEntry, type HistoryEntry } from "../lib/history";

function when(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Browser-local history of processed clips. Renders nothing when empty. */
export function HistoryPanel() {
  const [items, setItems] = useState<HistoryEntry[]>(() => loadHistory());
  if (items.length === 0) return null;

  return (
    <section className="mt-10">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-bay-ink">
          History <span className="num text-bay-ink-3">· {items.length}</span>
        </h2>
        <button
          className="btn px-2.5 py-1 text-xs"
          onClick={() => {
            clearHistory();
            setItems([]);
          }}
        >
          Clear all
        </button>
      </div>

      <p className="mb-3 text-xs text-bay-ink-3">
        Stored only in this browser — not uploaded, not synced across devices.
      </p>

      <ul className="space-y-2">
        {items.map((e) => (
          <li key={e.runId} className="panel flex items-center gap-3 p-3">
            <div className="h-12 w-20 shrink-0 overflow-hidden rounded bg-black/60">
              {e.thumbnail ? (
                <img src={e.thumbnail} alt="" className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-[10px] text-bay-ink-3">
                  {e.source === "url" ? "URL" : "clip"}
                </div>
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="truncate text-sm text-bay-ink" title={e.label}>
                {e.label}
              </div>
              <div className="num text-xs text-bay-ink-3">{when(e.time)}</div>
              {e.captions[0]?.text && (
                <div className="truncate text-xs text-bay-ink-2">{e.captions[0].text}</div>
              )}
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <Link to={`/results/${e.runId}`} className="btn px-3 py-1.5 text-xs">
                Open
              </Link>
              <button
                className="btn px-2.5 py-1.5 text-xs"
                aria-label="Delete from history"
                onClick={() => setItems(removeEntry(e.runId))}
              >
                ✕
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
