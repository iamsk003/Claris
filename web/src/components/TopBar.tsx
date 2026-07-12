import { Link } from "../router";
import { DEMO_RUN_ID } from "../demo/sampleRun";

export function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={"inline-flex items-center gap-2 " + className}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <rect x="2.5" y="4.5" width="19" height="15" rx="2.5" stroke="#9ba1a8" strokeWidth="1.3" />
        <line x1="14" y1="3.5" x2="14" y2="20.5" stroke="#ffb03a" strokeWidth="1.6" />
        <circle cx="8" cy="12" r="2.4" stroke="#9ba1a8" strokeWidth="1.3" />
      </svg>
      <span className="text-[15px] font-semibold tracking-tight text-bay-ink">CLARIS AI</span>
    </span>
  );
}

export function TopBar() {
  return (
    <header className="sticky top-0 z-30 border-b border-bay-line bg-bay-bg/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="rounded">
          <Logo />
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          <Link to="/upload" className="btn px-3 py-1.5">
            Upload
          </Link>
          <Link to={`/results/${DEMO_RUN_ID}`} className="btn btn-signal px-3 py-1.5">
            View sample
          </Link>
        </nav>
      </div>
    </header>
  );
}
