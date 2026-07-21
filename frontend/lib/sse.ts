// Consumes the chat streaming endpoint. Native EventSource can't POST or send an Authorization
// header, so we use fetch + a ReadableStream reader and parse SSE frames by hand.
//
// Wire format (from api/routes/chat_stream.py):
//   token: `data: {"token":"…"}\n\n`                        (default event, no `event:` line)
//   done:  `event: done\ndata: {"message_id":"…","citations":[…]}\n\n`
//   error: `event: error\ndata: {"error":"…","message":"…"}\n\n`
// A missing/other-user chat is a 404 JSON body *before* the stream — so check res.ok first.

import { API_BASE, ApiError, getToken } from "@/lib/api";
import type { ApiErrorBody, Citation } from "@/lib/types";

export interface StreamDone {
  message_id: string;
  citations: Citation[];
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (done: StreamDone) => void;
  onError: (err: ApiError) => void;
  signal?: AbortSignal;
}

export async function streamMessage(
  chatId: string,
  query: string,
  handlers: StreamHandlers,
): Promise<void> {
  const token = getToken();
  try {
    const res = await fetch(`${API_BASE}/v1/chats/${chatId}/messages/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ query }),
      signal: handlers.signal,
    });

    if (!res.ok || !res.body) {
      // Pre-stream failure (e.g. 404 unknown chat) arrives as a JSON error body.
      let body: Partial<ApiErrorBody> = {};
      try {
        body = (await res.json()) as Partial<ApiErrorBody>;
      } catch {
        // ignore non-JSON body
      }
      handlers.onError(
        new ApiError(body.error ?? "http_error", body.message ?? `Request failed (${res.status})`, res.status),
      );
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        dispatchFrame(frame, handlers);
      }
    }
  } catch (err) {
    // Stop button aborts the fetch — a clean end, not an error.
    if (err instanceof DOMException && err.name === "AbortError") return;
    handlers.onError(new ApiError("stream_error", err instanceof Error ? err.message : "Stream failed", 0));
  }
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;

  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return;
  }

  if (event === "done") {
    handlers.onDone(parsed as StreamDone);
  } else if (event === "error") {
    const e = parsed as ApiErrorBody;
    handlers.onError(new ApiError(e.error ?? "generation_error", e.message ?? "Generation failed", 200));
  } else {
    const t = parsed as { token?: string };
    if (typeof t.token === "string") handlers.onToken(t.token);
  }
}
