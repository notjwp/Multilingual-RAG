"use client";

import { MessageSquarePlusIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { useChats } from "@/lib/chats";

export default function AppIndexPage() {
  const { chats, loading, create } = useChats();
  const router = useRouter();

  // With existing chats, land on the most recent one (list is newest-first).
  useEffect(() => {
    if (!loading && chats.length > 0) {
      router.replace(`/chat/${chats[0].session_id}`);
    }
  }, [loading, chats, router]);

  async function startChat() {
    try {
      const chat = await create();
      router.push(`/chat/${chat.session_id}`);
    } catch {
      toast.error("Could not create a new chat.");
    }
  }

  if (loading || chats.length > 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-5 p-8 text-center">
      <div className="space-y-1">
        <h1 className="text-xl font-medium">Start a conversation</h1>
        <p className="text-sm text-muted-foreground">
          Ask questions and get grounded, cited answers from your documents.
        </p>
      </div>
      <Button onClick={startChat} className="gap-2">
        <MessageSquarePlusIcon className="size-4" />
        New chat
      </Button>
    </div>
  );
}
