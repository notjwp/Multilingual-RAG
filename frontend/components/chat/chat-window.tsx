"use client";

import { RotateCcwIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Composer } from "@/components/chat/composer";
import { MessageBubble, type UiMessage } from "@/components/chat/message-bubble";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import * as api from "@/lib/api";
import { useChats } from "@/lib/chats";
import { streamMessage } from "@/lib/sse";
import type { Message } from "@/lib/types";

function toUiMessage(m: Message): UiMessage {
  return {
    key: m.message_id,
    serverId: m.message_id,
    role: m.role,
    content: m.content,
    citations: m.citations,
  };
}

// Mounted with `key={chatId}` so switching chats remounts it with fresh state.
export function ChatWindow({ chatId }: { chatId: string }) {
  const { refresh } = useChats();
  const router = useRouter();
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load the chat's history.
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const detail = await api.getChat(chatId);
        if (active) setMessages(detail.messages.map(toUiMessage));
      } catch (err) {
        if (!active) return;
        if (err instanceof api.ApiError && err.status === 404) router.replace("/");
        else toast.error("Could not load this chat.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [chatId, router]);

  // Keep the newest content in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Stream one query into a specific assistant bubble (shared by first send and retry).
  async function runStream(query: string, assistantKey: string, wasFirstTurn: boolean) {
    setStreaming(true);
    const controller = new AbortController();
    controllerRef.current = controller;
    let completed = false;

    await streamMessage(chatId, query, {
      signal: controller.signal,
      onToken: (text) =>
        setMessages((prev) =>
          prev.map((m) => (m.key === assistantKey ? { ...m, content: m.content + text } : m)),
        ),
      onDone: ({ message_id, citations }) => {
        completed = true;
        setMessages((prev) =>
          prev.map((m) =>
            m.key === assistantKey
              ? { ...m, serverId: message_id, citations, pending: false, error: false }
              : m,
          ),
        );
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((m) => (m.key === assistantKey ? { ...m, pending: false, error: true } : m)),
        );
        toast.error(err.message);
      },
    });

    // Stream finished (done, error, or aborted via Stop).
    setStreaming(false);
    controllerRef.current = null;
    setMessages((prev) =>
      prev.map((m) => (m.key === assistantKey && m.pending ? { ...m, pending: false } : m)),
    );
    // A fresh chat gets its title from the first message — refresh the sidebar to show it.
    if (completed && wasFirstTurn) void refresh();
  }

  async function send(query: string) {
    if (streaming) return;
    const wasFirstTurn = messages.length === 0;
    const assistantKey = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { key: crypto.randomUUID(), role: "user", content: query, citations: [] },
      { key: assistantKey, role: "assistant", content: "", citations: [], pending: true },
    ]);
    await runStream(query, assistantKey, wasFirstTurn);
  }

  // Re-run the last user turn into the failed assistant bubble (a failed stream isn't persisted
  // server-side, so this creates no duplicate turn).
  function retry() {
    if (streaming) return;
    const n = messages.length;
    const last = messages[n - 1];
    const prevUser = messages[n - 2];
    if (!last || last.role !== "assistant" || !last.error) return;
    if (!prevUser || prevUser.role !== "user") return;
    setMessages((prev) =>
      prev.map((m) => (m.key === last.key ? { ...m, content: "", error: false, pending: true } : m)),
    );
    void runStream(prevUser.content, last.key, n <= 2);
  }

  function stop() {
    controllerRef.current?.abort();
  }

  const lastMessage = messages.at(-1);
  const canRetry = !loading && !streaming && lastMessage?.role === "assistant" && !!lastMessage.error;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-6">
          {loading ? (
            <>
              <Skeleton className="h-16 w-2/3" />
              <Skeleton className="ml-auto h-10 w-1/2" />
              <Skeleton className="h-20 w-3/4" />
            </>
          ) : messages.length === 0 ? (
            <div className="flex flex-col items-center gap-1 py-16 text-center">
              <p className="text-base font-medium">Ask anything</p>
              <p className="text-sm text-muted-foreground">
                Answers are grounded in your uploaded documents, with citations.
              </p>
            </div>
          ) : (
            messages.map((m) => <MessageBubble key={m.key} message={m} />)
          )}
          {canRetry && (
            <div className="flex justify-start">
              <Button variant="outline" size="sm" className="gap-1.5" onClick={retry}>
                <RotateCcwIcon className="size-3.5" />
                Retry
              </Button>
            </div>
          )}
        </div>
      </div>

      <div className="border-t bg-background">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <Composer onSend={send} onStop={stop} streaming={streaming} disabled={loading} />
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Multilingual RAG can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}
