"use client";

import { useParams } from "next/navigation";

import { ChatWindow } from "@/components/chat/chat-window";

export default function ChatPage() {
  const params = useParams();
  const chatId = typeof params.chatId === "string" ? params.chatId : "";
  if (!chatId) return null;
  // key remounts the window (fresh state) when switching between chats.
  return <ChatWindow key={chatId} chatId={chatId} />;
}
