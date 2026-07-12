import { motion } from "framer-motion";
import { TopBar, Logo } from "../components/TopBar";
import { PipelineDiagram } from "../components/PipelineDiagram";
import { Link } from "../router";
import { DEMO_RUN_ID } from "../demo/sampleRun";

const FEATURES = [
  {
    title: "Grounded, not guessed",
    body: "Every caption is checked claim-by-claim against a timestamped evidence ledger. Ungrounded lines are regenerated, not shipped.",
  },
  {
    title: "Four voices, one clip",
    body: "Formal, sarcastic, tech humor, and everyday humor — each scored for tone and forced to stay distinct from the others.",
  },
  {
    title: "Traceable to the frame",
    body: "Hover any sentence and the exact speech, text, audio, and visual evidence behind it lights up on the timeline.",
  },
];

const TECH = ["Whisper", "PaddleOCR", "OpenCV", "Fireworks AI"];

const stagger = {
  hidden: { opacity: 0, y: 10 },
  show: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.06, duration: 0.4 } }),
};

export function Landing() {
  return (
    <div className="min-h-full">
      <TopBar />

      {/* Hero */}
      <section className="mx-auto max-w-7xl px-6 pb-10 pt-16 sm:pt-24">
        <motion.div initial="hidden" animate="show" className="max-w-3xl">
          <motion.div variants={stagger} custom={0} className="num mb-4 inline-flex items-center gap-2 text-xs uppercase tracking-widest text-bay-ink-3">
            <span className="h-1.5 w-1.5 rounded-full bg-signal" /> multimodal video captioning
          </motion.div>
          <motion.h1
            variants={stagger}
            custom={1}
            className="text-4xl font-semibold leading-[1.1] tracking-tight text-bay-ink sm:text-6xl"
          >
            Captions you can <span className="text-signal">trace back</span> to the footage.
          </motion.h1>
          <motion.p variants={stagger} custom={2} className="mt-5 max-w-2xl text-lg text-bay-ink-2">
            CLARIS AI turns a short clip into four styled captions and shows its work — each line
            linked to the timestamped speech, on-screen text, audio, and visuals that support it.
          </motion.p>
          <motion.div variants={stagger} custom={3} className="mt-8 flex flex-wrap items-center gap-3">
            <Link to="/upload" className="btn btn-signal px-5 py-2.5 text-base">
              Upload a clip
            </Link>
            <Link to={`/results/${DEMO_RUN_ID}`} className="btn px-5 py-2.5 text-base">
              Watch the sample
            </Link>
          </motion.div>
        </motion.div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="grid gap-3 sm:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05, duration: 0.4 }}
              className="panel p-5"
            >
              <h3 className="text-base font-semibold text-bay-ink">{f.title}</h3>
              <p className="mt-2 text-sm text-bay-ink-2">{f.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-bay-ink">How it works</h2>
          <p className="mt-1 text-sm text-bay-ink-2">
            One pass, thirteen stages. Perception builds the evidence, generation drafts
            candidates, verification gates them.
          </p>
        </div>
        <div className="panel p-4">
          <PipelineDiagram />
        </div>
      </section>

      {/* Technologies */}
      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="flex flex-wrap items-center gap-2">
          <span className="num mr-2 text-xs uppercase tracking-widest text-bay-ink-3">Built with</span>
          {TECH.map((t) => (
            <span key={t} className="chip">
              {t}
            </span>
          ))}
        </div>
      </section>

      <footer className="mt-8 border-t border-bay-line">
        <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-3 px-6 py-8 sm:flex-row sm:items-center">
          <Logo />
          <p className="num text-xs text-bay-ink-3">
            Grounded multi-style video captioning · MIT licensed
          </p>
        </div>
      </footer>
    </div>
  );
}
