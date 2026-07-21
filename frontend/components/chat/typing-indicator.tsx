"use client";

import { motion, useReducedMotion } from "framer-motion";

// Three bouncing dots shown in the assistant bubble until the first token arrives.
export function TypingIndicator() {
  const reduce = useReducedMotion();
  return (
    <div className="flex items-center gap-1 py-1" role="status" aria-label="Assistant is typing">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="size-1.5 rounded-full bg-current opacity-60"
          animate={reduce ? undefined : { opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.15, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}
