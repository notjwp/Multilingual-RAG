"use client";

import { ThemeProvider } from "next-themes";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/lib/auth";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <AuthProvider>
        <TooltipProvider>{children}</TooltipProvider>
        {/* Inside ThemeProvider so Sonner picks up the active theme. */}
        <Toaster richColors position="top-center" />
      </AuthProvider>
    </ThemeProvider>
  );
}
