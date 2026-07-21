"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { BackgroundGrid } from "@/components/background-grid";
import { useAuth } from "@/lib/auth";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  // Already signed in → no reason to see login/signup.
  useEffect(() => {
    if (!loading && user) router.replace("/");
  }, [loading, user, router]);

  return (
    <div className="relative flex min-h-dvh items-center justify-center overflow-hidden p-4">
      <BackgroundGrid />
      <div className="relative z-10 w-full max-w-sm">{children}</div>
    </div>
  );
}
