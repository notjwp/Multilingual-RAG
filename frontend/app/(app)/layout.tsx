import { Sidebar } from "@/components/chat/sidebar";
import { RequireAuth } from "@/lib/auth";
import { ChatsProvider } from "@/lib/chats";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <RequireAuth>
      <ChatsProvider>
        <div className="flex h-dvh">
          <Sidebar />
          <main className="flex min-w-0 flex-1 flex-col">{children}</main>
        </div>
      </ChatsProvider>
    </RequireAuth>
  );
}
