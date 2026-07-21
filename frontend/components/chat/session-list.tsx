"use client";

import { MoreHorizontalIcon, PencilIcon, Trash2Icon } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useChats } from "@/lib/chats";
import { cn } from "@/lib/utils";

export function SessionList() {
  const { chats, loading, rename, remove } = useChats();
  const params = useParams();
  const activeId = typeof params.chatId === "string" ? params.chatId : undefined;
  const router = useRouter();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus + select the rename field once it renders (after the menu closes).
  useEffect(() => {
    if (!editingId) return;
    const raf = requestAnimationFrame(() => inputRef.current?.select());
    return () => cancelAnimationFrame(raf);
  }, [editingId]);

  function startEdit(id: string, title: string) {
    setEditValue(title);
    setEditingId(id);
  }

  async function commitEdit(id: string) {
    const title = editValue.trim();
    setEditingId(null);
    if (!title) return;
    try {
      await rename(id, title);
    } catch {
      toast.error("Could not rename chat.");
    }
  }

  async function confirmDelete() {
    const id = deleteId;
    setDeleteId(null);
    if (!id) return;
    try {
      await remove(id);
      if (activeId === id) router.replace("/");
    } catch {
      toast.error("Could not delete chat.");
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-1 px-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    );
  }

  if (chats.length === 0) {
    return <p className="px-4 py-2 text-sm text-muted-foreground">No chats yet.</p>;
  }

  const pendingDelete = chats.find((c) => c.session_id === deleteId);

  return (
    <>
      <nav className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3">
        {chats.map((chat) => {
          const active = chat.session_id === activeId;

          if (editingId === chat.session_id) {
            return (
              <Input
                key={chat.session_id}
                ref={inputRef}
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void commitEdit(chat.session_id);
                  } else if (e.key === "Escape") {
                    setEditingId(null);
                  }
                }}
                onBlur={() => setEditingId(null)}
                className="h-8"
              />
            );
          }

          return (
            <div
              key={chat.session_id}
              className={cn(
                "group/row flex items-center gap-1 rounded-lg pr-1 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                active && "bg-sidebar-accent text-sidebar-accent-foreground",
              )}
            >
              <Link
                href={`/chat/${chat.session_id}`}
                className="min-w-0 flex-1 truncate px-2 py-2 text-sm"
                title={chat.title}
              >
                {chat.title}
              </Link>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Chat options"
                      className="shrink-0 opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100 aria-expanded:opacity-100"
                    />
                  }
                >
                  <MoreHorizontalIcon className="size-4" />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => startEdit(chat.session_id, chat.title)}>
                    <PencilIcon className="size-4" />
                    Rename
                  </DropdownMenuItem>
                  <DropdownMenuItem variant="destructive" onClick={() => setDeleteId(chat.session_id)}>
                    <Trash2Icon className="size-4" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          );
        })}
      </nav>

      <Dialog open={deleteId !== null} onOpenChange={(open) => !open && setDeleteId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete chat?</DialogTitle>
            <DialogDescription>
              {pendingDelete ? `"${pendingDelete.title}" and its messages` : "This chat and its messages"}{" "}
              will be permanently deleted. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline">Cancel</Button>} />
            <Button variant="destructive" onClick={() => void confirmDelete()}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
