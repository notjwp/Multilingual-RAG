"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import * as api from "@/lib/api";
import type { ChatSession } from "@/lib/types";

interface ChatsState {
  chats: ChatSession[];
  loading: boolean;
  refresh: () => Promise<void>;
  create: (title?: string) => Promise<ChatSession>;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const ChatsContext = createContext<ChatsState | null>(null);

// Client-side cache of the user's chat sessions, shared by the sidebar and the chat window so
// mutations (create/rename/delete) and the post-first-message title refresh update in place
// without a full reload.
export function ChatsProvider({ children }: { children: React.ReactNode }) {
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    api
      .listChats()
      .then((list) => {
        if (active) setChats(list);
      })
      .catch(() => {
        // apiFetch already handles auth failures; leave the list empty otherwise.
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    setChats(await api.listChats());
  }, []);

  const create = useCallback(async (title?: string) => {
    const chat = await api.createChat(title);
    setChats((prev) => [chat, ...prev]);
    return chat;
  }, []);

  const rename = useCallback(async (id: string, title: string) => {
    const updated = await api.renameChat(id, title);
    setChats((prev) => prev.map((c) => (c.session_id === id ? updated : c)));
  }, []);

  const remove = useCallback(async (id: string) => {
    await api.deleteChat(id);
    setChats((prev) => prev.filter((c) => c.session_id !== id));
  }, []);

  const value = useMemo<ChatsState>(
    () => ({ chats, loading, refresh, create, rename, remove }),
    [chats, loading, refresh, create, rename, remove],
  );

  return <ChatsContext.Provider value={value}>{children}</ChatsContext.Provider>;
}

export function useChats(): ChatsState {
  const ctx = useContext(ChatsContext);
  if (!ctx) throw new Error("useChats must be used within a ChatsProvider");
  return ctx;
}
