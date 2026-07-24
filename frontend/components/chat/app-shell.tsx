"use client";

import { MenuIcon } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Sidebar } from "@/components/chat/sidebar";
import { CommandPalette } from "@/components/command-palette";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { useChats } from "@/lib/chats";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { create } = useChats();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);

  // Close the mobile drawer on navigation (deferred so it isn't a synchronous effect setState).
  useEffect(() => {
    const id = window.setTimeout(() => setDrawerOpen(false), 0);
    return () => window.clearTimeout(id);
  }, [pathname]);

  // Global shortcuts: ⌘K / Ctrl-K toggles the palette; ⌘⇧O starts a new chat.
  useEffect(() => {
    async function newChat() {
      try {
        const chat = await create();
        router.push(`/chat/${chat.session_id}`);
      } catch {
        toast.error("Could not create a new chat.");
      }
    }
    function onKey(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((open) => !open);
      } else if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        void newChat();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [create, router]);

  // Let any component (e.g. the sidebar Search button) open the palette without prop-drilling.
  useEffect(() => {
    function open() {
      setPaletteOpen(true);
    }
    window.addEventListener("open-command-palette", open);
    return () => window.removeEventListener("open-command-palette", open);
  }, []);

  return (
    <div className="flex h-dvh">
      <div className="hidden md:flex">
        <Sidebar />
      </div>

      <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
        <SheetContent>
          <Sidebar />
        </SheetContent>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b p-2 md:hidden">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Open menu"
            onClick={() => setDrawerOpen(true)}
          >
            <MenuIcon className="size-5" />
          </Button>
          <span className="font-heading text-sm font-medium">Multilingual RAG</span>
        </header>
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</main>
      </div>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}
