"use client";

import { motion, useReducedMotion } from "framer-motion";

// Decorative backdrop for the auth pages: a subtle dot grid plus two slowly drifting
// gradient blobs. Purely presentational (aria-hidden) and motion is disabled when the
// user prefers reduced motion.
export function BackgroundGrid() {
  const reduce = useReducedMotion();

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle,var(--border)_1px,transparent_1px)] [background-size:22px_22px] opacity-40 [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]" />
      <motion.div
        className="absolute -top-24 left-1/4 h-72 w-72 rounded-full bg-primary/20 blur-3xl"
        animate={reduce ? undefined : { x: [0, 40, 0], y: [0, 30, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute right-1/4 bottom-0 h-72 w-72 rounded-full bg-sky-500/20 blur-3xl"
        animate={reduce ? undefined : { x: [0, -30, 0], y: [0, -20, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}
