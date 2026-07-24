import { AppShell } from "@/components/chat/app-shell";
import { RequireAuth } from "@/lib/auth";
import { ChatsProvider } from "@/lib/chats";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <RequireAuth>
      <ChatsProvider>
        <AppShell>{children}</AppShell>
      </ChatsProvider>
    </RequireAuth>
  );
}
