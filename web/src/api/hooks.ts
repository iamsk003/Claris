import { useMutation, useQuery } from "@tanstack/react-query";
import { getResult, startRun, uploadClip } from "./client";
import { getDemoResult, isDemoRunId } from "../demo/sampleRun";
import { getHistoryResult } from "../lib/history";

export function useUploadClip() {
  return useMutation({
    mutationFn: (args: { file: File; onProgress?: (f: number) => void }) =>
      uploadClip(args.file, args.onProgress),
  });
}

export function useStartRun() {
  return useMutation({ mutationFn: (clipId: string) => startRun(clipId) });
}

/**
 * The results envelope. A demo run id resolves to the bundled sample so a static Vercel
 * deploy (no backend) still renders the results page end to end.
 */
export function useRunResult(runId: string | undefined) {
  return useQuery({
    queryKey: ["run", runId],
    enabled: !!runId,
    retry: 1,
    queryFn: async () => {
      if (!runId) throw new Error("missing runId");
      if (isDemoRunId(runId)) return getDemoResult();
      try {
        return await getResult(runId);
      } catch (e) {
        // Reopen from browser-local history when the backend no longer has the run.
        const cached = getHistoryResult(runId);
        if (cached) return cached;
        throw e;
      }
    },
  });
}
