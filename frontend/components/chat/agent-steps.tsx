"use client";

// The agent's working notes, shown live above an answer and collapsed once it lands.
//
// Mirrors SourcesList (citation-chip.tsx): a compact list inside the assistant bubble, sources
// below the answer, steps above it. Steps are ephemeral — they arrive only over SSE and are never
// persisted, so a reloaded chat renders none of this.

import { CheckIcon, ChevronRightIcon, Loader2Icon } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";

import type { AgentStep } from "@/lib/types";
import { cn } from "@/lib/utils";

export function AgentSteps({ steps, pending }: { steps: AgentStep[]; pending?: boolean }) {
  const reduce = useReducedMotion();
  const [expanded, setExpanded] = useState(false);
  if (steps.length === 0) return null;

  // While the answer is still streaming the list is always open — that is the point of it.
  const open = pending || expanded;

  return (
    <div className="mb-2 border-b pb-2">
      {!pending && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <ChevronRightIcon
            className={cn("size-3 transition-transform", expanded && "rotate-90")}
          />
          {summarize(steps)}
        </button>
      )}
      {open && (
        <ol className={cn("flex flex-col gap-1", !pending && "mt-1.5")}>
          {steps.map((step) => (
            <motion.li
              key={step.id}
              initial={reduce ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="flex items-center gap-1.5 text-xs"
            >
              {step.status === "running" ? (
                <Loader2Icon className="size-3 shrink-0 animate-spin text-muted-foreground" />
              ) : (
                <CheckIcon className="size-3 shrink-0 text-muted-foreground" />
              )}
              <span className="text-muted-foreground">{step.label}</span>
              {step.detail && (
                <span className="min-w-0 truncate text-muted-foreground/70">· {step.detail}</span>
              )}
            </motion.li>
          ))}
        </ol>
      )}
    </div>
  );
}

// The collapsed one-liner. Surfaces the two things worth knowing at a glance: how much work it
// did, and whether it had to retry.
function summarize(steps: AgentStep[]): string {
  const searches = steps.filter((s) => s.node === "retrieve" && s.status === "done").length;
  const parts = [`Thought for ${steps.length} step${steps.length === 1 ? "" : "s"}`];
  if (searches > 1) parts.push(`${searches} searches`);
  const routed = steps.find((s) => s.node === "route_language" && s.detail);
  if (routed?.detail) parts.push(routed.detail);
  return parts.join(" · ");
}
