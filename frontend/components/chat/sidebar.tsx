"use client";

import { LogOutIcon, PlusIcon, SearchIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { SessionList } from "@/components/chat/session-list";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { useChats } from "@/lib/chats";

export function Sidebar() {
  const router = useRouter();
  const { create } = useChats();
  const { user, logout } = useAuth();

  async function onNewChat() {
    try {
      const chat = await create();
      router.push(`/chat/${chat.session_id}`);
    } catch {
      toast.error("Could not create a new chat.");
    }
  }

  function onLogout() {
    logout();
    router.replace("/login");
  }

  return (
    <aside className="flex h-dvh w-72 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground">
      <div className="flex flex-col gap-1 p-3">
        <Button onClick={onNewChat} variant="outline" className="w-full justify-start gap-2">
          <PlusIcon className="size-4" />
          New chat
        </Button>
        <button
          type="button"
          onClick={() => window.dispatchEvent(new CustomEvent("open-command-palette"))}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        >
          <SearchIcon className="size-4" />
          <span className="flex-1 text-left">Search</span>
          <kbd className="rounded border px-1.5 py-0.5 text-[10px] text-muted-foreground">⌘K</kbd>
        </button>
      </div>

      <SessionList />

      <div className="mt-auto border-t p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="min-w-0 truncate text-xs text-muted-foreground" title={user?.email}>
            {user?.email}
          </span>
          <div className="flex shrink-0 items-center gap-1">
            <ThemeToggle />
            <Button variant="ghost" size="icon-sm" onClick={onLogout} aria-label="Log out">
              <LogOutIcon className="size-4" />
            </Button>
          </div>
        </div>
      </div>
    </aside>
  );
}
