"use client";

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import * as api from "@/lib/api";
import type { User } from "@/lib/types";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Validate a persisted token on mount so a refresh keeps the session (or clears a stale token).
  useEffect(() => {
    let active = true;
    (async () => {
      let nextUser: User | null = null;
      const token = api.getToken();
      if (token) {
        try {
          nextUser = await api.me();
        } catch {
          api.clearToken();
        }
      } else {
        // Cross an async boundary so these state updates aren't synchronous with the effect.
        await Promise.resolve();
      }
      if (active) {
        setUser(nextUser);
        setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login(email, password);
    api.setToken(res.access_token);
    setUser(res.user);
  }, []);

  const signup = useCallback(async (email: string, password: string) => {
    const res = await api.signup(email, password);
    api.setToken(res.access_token);
    setUser(res.user);
  }, []);

  const logout = useCallback(() => {
    api.clearToken();
    setUser(null);
  }, []);

  // Keep the session alive: refresh the short-lived token periodically while signed in.
  // (A lapse just means the next protected request 401s and RequireAuth routes to /login.)
  useEffect(() => {
    if (!user) return;
    const interval = setInterval(
      () => {
        api
          .refresh()
          .then((res) => api.setToken(res.access_token))
          .catch(() => undefined);
      },
      20 * 60 * 1000,
    );
    return () => clearInterval(interval);
  }, [user]);

  const value = useMemo<AuthState>(
    () => ({ user, loading, login, signup, logout }),
    [user, loading, login, signup, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}

// Wraps the authenticated app group; redirects to /login when there is no session.
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="flex h-dvh items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    );
  }
  return <>{children}</>;
}
