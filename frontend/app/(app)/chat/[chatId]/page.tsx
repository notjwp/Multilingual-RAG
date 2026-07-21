"use client";

import { useParams } from "next/navigation";

// Placeholder — Part 3 replaces this with the streaming chat window.
export default function ChatPage() {
  const params = useParams();
  const chatId = typeof params.chatId === "string" ? params.chatId : "";
  return (
    <div className="flex flex-1 items-center justify-center p-8 text-center text-sm text-muted-foreground">
      Chat <span className="mx-1 font-mono text-xs">{chatId}</span> — the streaming window arrives in
      Part 3.
    </div>
  );
}
