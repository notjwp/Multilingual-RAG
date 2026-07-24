"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import {
  type LucideIcon,
  LogOutIcon,
  MessageSquareIcon,
  MessageSquarePlusIcon,
  MoonIcon,
  SearchIcon,
  SunIcon,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { useAuth } from "@/lib/auth";
import { useChats } from "@/lib/chats";
import { cn } from "@/lib/utils";

interface CommandItem {
  id: string;
  label: string;
  icon: LucideIcon;
  run: () => void | Promise<void>;
}

export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const { chats, create } = useChats();
  const { logout } = useAuth();
  const { resolvedTheme, setTheme } = useTheme();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);

  function close() {
    // Reset here (an event handler) rather than in an effect, to keep lint happy.
    setQuery("");
    setActive(0);
    onOpenChange(false);
  }

  const items = useMemo<CommandItem[]>(() => {
    const actions: CommandItem[] = [
      {
        id: "new-chat",
        label: "New chat",
        icon: MessageSquarePlusIcon,
        run: async () => {
          try {
            const chat = await create();
            router.push(`/chat/${chat.session_id}`);
          } catch {
            toast.error("Could not create a new chat.");
          }
        },
      },
      {
        id: "theme",
        label: "Toggle theme",
        icon: resolvedTheme === "dark" ? SunIcon : MoonIcon,
        run: () => setTheme(resolvedTheme === "dark" ? "light" : "dark"),
      },
      {
        id: "logout",
        label: "Log out",
        icon: LogOutIcon,
        run: () => {
          logout();
          router.replace("/login");
        },
      },
    ];
    const chatItems: CommandItem[] = chats.map((chat) => ({
      id: `chat-${chat.session_id}`,
      label: chat.title,
      icon: MessageSquareIcon,
      run: () => router.push(`/chat/${chat.session_id}`),
    }));

    const q = query.trim().toLowerCase();
    const all = [...actions, ...chatItems];
    return q ? all.filter((item) => item.label.toLowerCase().includes(q)) : all;
  }, [query, chats, create, router, logout, resolvedTheme, setTheme]);

  function activate(item: CommandItem | undefined) {
    if (!item) return;
    close();
    void item.run();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      activate(items[active]);
    }
  }

  return (
    <DialogPrimitive.Root open={open} onOpenChange={(next) => (next ? onOpenChange(true) : close())}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop className="fixed inset-0 z-50 bg-black/40 transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0" />
        <DialogPrimitive.Popup className="fixed top-[18%] left-1/2 z-50 w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 overflow-hidden rounded-xl bg-popover text-popover-foreground shadow-lg ring-1 ring-foreground/10 outline-none transition-all duration-150 data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0">
          <div className="flex items-center gap-2 border-b px-3">
            <SearchIcon className="size-4 shrink-0 text-muted-foreground" />
            <input
              autoFocus
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              onKeyDown={onKeyDown}
              placeholder="Search chats or run a command…"
              className="h-11 w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>
          <ul className="max-h-80 overflow-y-auto p-1">
            {items.length === 0 ? (
              <li className="px-3 py-6 text-center text-sm text-muted-foreground">No results</li>
            ) : (
              items.map((item, i) => {
                const Icon = item.icon;
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      onMouseMove={() => setActive(i)}
                      onClick={() => activate(item)}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm",
                        i === active && "bg-accent text-accent-foreground",
                      )}
                    >
                      <Icon className="size-4 shrink-0 text-muted-foreground" />
                      <span className="truncate">{item.label}</span>
                    </button>
                  </li>
                );
              })
            )}
          </ul>
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
