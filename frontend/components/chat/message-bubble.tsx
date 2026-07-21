"use client";

import { motion, useReducedMotion } from "framer-motion";

import { SourcesList } from "@/components/chat/citation-chip";
import { Markdown } from "@/components/chat/markdown";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import type { Citation, Role } from "@/lib/types";
import { cn } from "@/lib/utils";

// UI-side message model. `key` is a stable client id (unchanged across the streamed→persisted id
// swap, so the bubble doesn't remount); `serverId` is the backend message_id once known.
export interface UiMessage {
  key: string;
  serverId?: string;
  role: Role;
  content: string;
  citations: Citation[];
  pending?: boolean;
  error?: boolean;
}

export function MessageBubble({ message }: { message: UiMessage }) {
  const reduce = useReducedMotion();
  const isUser = message.role === "user";
  const showTyping = message.role === "assistant" && message.pending && !message.content;

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn("flex", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-2.5 text-sm",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground",
        )}
      >
        {showTyping ? (
          <TypingIndicator />
        ) : isUser ? (
          <div className="whitespace-pre-wrap break-words leading-relaxed">{message.content}</div>
        ) : (
          <>
            <Markdown content={message.content} citations={message.citations} />
            {!message.pending && <SourcesList citations={message.citations} />}
          </>
        )}
        {message.error && (
          <p className="mt-1 text-xs text-destructive">Couldn’t generate a response.</p>
        )}
      </div>
    </motion.div>
  );
}
