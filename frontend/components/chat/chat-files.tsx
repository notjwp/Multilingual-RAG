"use client";

import {
  AlertCircleIcon,
  FileTextIcon,
  Loader2Icon,
  PaperclipIcon,
  Trash2Icon,
  UploadCloudIcon,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import * as api from "@/lib/api";
import type { DocumentItem, IngestionJob } from "@/lib/types";
import { cn } from "@/lib/utils";

interface UploadEntry {
  key: string;
  name: string;
  phase: "uploading" | "queued" | "indexing" | "failed";
  hint?: string;
  error?: string;
}

// Stored source is "<32-hex>_<original name>" under a path — show just the original file name.
function cleanName(source: string): string {
  const base = source.split(/[\\/]/).pop() ?? source;
  return base.replace(/^[0-9a-f]{32}_/i, "");
}

// Poll until the backend reports a terminal state. Big files on CPU can take minutes, so we never
// declare failure on our own before a long cap — a slow job is not a failed one. The callback lets
// the UI reflect queued-vs-indexing and surface a hint if the job is stuck queued (no worker).
const MAX_POLL_MS = 30 * 60 * 1000;

async function pollJob(
  jobId: string,
  onStatus: (status: string, elapsedMs: number) => void,
): Promise<IngestionJob> {
  const start = Date.now();
  let delay = 1200;
  while (Date.now() - start < MAX_POLL_MS) {
    const job = await api.getIngestionJob(jobId);
    onStatus(job.status, Date.now() - start);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, delay));
    delay = Math.min(delay + 300, 3000);
  }
  return { job_id: jobId, status: "failed", error_message: "Indexing timed out." };
}

// Per-chat file drawer: documents uploaded here ground only THIS chat's answers, and are invisible
// to every other chat (backend scopes retrieval by chat).
export function ChatFiles({ chatId }: { chatId: string }) {
  const [open, setOpen] = useState(false);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [uploads, setUploads] = useState<UploadEntry[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load this chat's documents once (also drives the header count).
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const list = await api.listChatDocuments(chatId);
        if (active) setDocs(list);
      } catch {
        /* leave empty; opening the drawer can retry */
      } finally {
        if (active) setLoaded(true);
      }
    })();
    return () => {
      active = false;
    };
  }, [chatId]);

  async function refresh() {
    try {
      setDocs(await api.listChatDocuments(chatId));
    } catch {
      /* keep the current list on a transient failure */
    }
  }

  async function handleFiles(files: FileList | File[]) {
    for (const file of Array.from(files)) {
      const key = crypto.randomUUID();
      setUploads((prev) => [...prev, { key, name: file.name, phase: "uploading" }]);
      try {
        const job = await api.uploadChatDocument(chatId, file);
        const final = await pollJob(job.job_id, (status, elapsedMs) => {
          setUploads((prev) =>
            prev.map((u) => {
              if (u.key !== key) return u;
              if (status === "queued") {
                return {
                  ...u,
                  phase: "queued",
                  // Stuck queued means nothing is picking it up — most likely no worker.
                  hint:
                    elapsedMs > 20000 ? "Waiting for the indexing worker — is it running?" : undefined,
                };
              }
              if (status === "running") {
                return {
                  ...u,
                  phase: "indexing",
                  hint: elapsedMs > 60000 ? "Large files can take a few minutes to embed." : undefined,
                };
              }
              return u;
            }),
          );
        });
        if (final.status === "succeeded") {
          setUploads((prev) => prev.filter((u) => u.key !== key));
          toast.success(`"${file.name}" added to this chat.`);
          await refresh();
        } else {
          setUploads((prev) =>
            prev.map((u) =>
              u.key === key ? { ...u, phase: "failed", error: final.error_message ?? undefined } : u,
            ),
          );
          toast.error(`"${file.name}" failed to index.`);
        }
      } catch (err) {
        setUploads((prev) => prev.map((u) => (u.key === key ? { ...u, phase: "failed" } : u)));
        toast.error(err instanceof api.ApiError ? err.message : `Could not upload "${file.name}".`);
      }
    }
  }

  async function remove(id: string) {
    setDocs((prev) => prev.filter((d) => d.document.document_id !== id)); // optimistic
    try {
      await api.deleteChatDocument(chatId, id);
      toast.success("Removed from this chat.");
    } catch {
      toast.error("Could not remove document.");
      await refresh();
    }
  }

  const count = docs.length;

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => setOpen(true)}
        aria-label="Files in this chat"
      >
        <PaperclipIcon className="size-4" />
        Files
        {count > 0 && (
          <span className="rounded bg-muted px-1.5 text-xs text-muted-foreground">{count}</span>
        )}
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Files in this chat</DialogTitle>
            <DialogDescription>
              Uploads here ground only this chat&apos;s answers — other chats can&apos;t see them.
            </DialogDescription>
          </DialogHeader>

          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files.length) void handleFiles(e.dataTransfer.files);
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed p-6 text-center transition-colors outline-none focus-visible:border-ring",
              dragOver ? "border-primary bg-primary/5" : "border-border hover:border-foreground/30",
            )}
          >
            <UploadCloudIcon className="size-6 text-muted-foreground" />
            <p className="text-sm font-medium">Drag files here, or click to browse</p>
            <p className="text-xs text-muted-foreground">TXT, Markdown, or PDF · up to 25 MB each</p>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".txt,.md,.markdown,.pdf"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) void handleFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          {uploads.length > 0 && (
            <ul className="flex flex-col gap-2">
              {uploads.map((u) => (
                <li
                  key={u.key}
                  className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3 py-2 text-sm"
                >
                  {u.phase === "failed" ? (
                    <AlertCircleIcon className="size-4 shrink-0 text-destructive" />
                  ) : (
                    <Loader2Icon className="size-4 shrink-0 animate-spin text-muted-foreground" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="truncate">{u.name}</p>
                    {(u.error || u.hint) && (
                      <p
                        className={cn(
                          "truncate text-xs",
                          u.phase === "failed" ? "text-destructive" : "text-muted-foreground",
                        )}
                      >
                        {u.error ?? u.hint}
                      </p>
                    )}
                  </div>
                  <span
                    className={cn(
                      "shrink-0 text-xs",
                      u.phase === "failed" ? "text-destructive" : "text-muted-foreground",
                    )}
                  >
                    {u.phase === "uploading"
                      ? "Uploading…"
                      : u.phase === "queued"
                        ? "Queued…"
                        : u.phase === "indexing"
                          ? "Indexing…"
                          : "Failed"}
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="max-h-64 overflow-y-auto">
            {!loaded ? (
              <div className="flex flex-col gap-2">
                {Array.from({ length: 2 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : docs.length === 0 ? (
              <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                No files yet. Upload one to ground this chat&apos;s answers.
              </p>
            ) : (
              <ul className="flex flex-col gap-2">
                {docs.map((item) => (
                  <li
                    key={item.document.document_id}
                    className="flex items-center gap-3 rounded-lg border px-3 py-2"
                  >
                    <FileTextIcon className="size-5 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">
                        {cleanName(item.document.source)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {item.chunk_count} chunk{item.chunk_count === 1 ? "" : "s"}
                        {item.document.language && item.document.language !== "unknown"
                          ? ` · ${item.document.language}`
                          : ""}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Remove ${cleanName(item.document.source)}`}
                      onClick={() => void remove(item.document.document_id)}
                    >
                      <Trash2Icon className="size-4" />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
