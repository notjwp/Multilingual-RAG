// Single typed seam for every backend call: injects the Bearer token, normalizes the
// `{error, message}` error contract into `ApiError`, and on a 401 for an *authenticated*
// request clears the token and bounces to /login.

import type {
  AuthResponse,
  ChatDetail,
  ChatSession,
  DocumentItem,
  IngestionJob,
  Message,
  User,
} from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const TOKEN_KEY = "mrag_token";

// Token storage — the one place that touches localStorage, so an M17 httpOnly-cookie swap
// (or Part 5) only edits here.
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(code: string, message: string, status: number, details?: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let code = "http_error";
  let message = res.statusText || `Request failed (${res.status})`;
  let details: Record<string, unknown> | undefined;
  try {
    const body = (await res.json()) as { error?: string; message?: string; details?: Record<string, unknown> };
    if (body && typeof body === "object") {
      code = body.error ?? code;
      message = body.message ?? message;
      details = body.details;
    }
  } catch {
    // non-JSON error body — keep the status-line defaults
  }
  return new ApiError(code, message, res.status, details);
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });

  if (!res.ok) {
    const err = await parseError(res);
    // A 401 on a request we *authenticated* means the session expired → sign out and redirect.
    // A 401 without a token (e.g. bad login credentials) is surfaced to the caller instead.
    if (res.status === 401 && token) {
      clearToken();
      if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
    throw err;
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// --- Auth ---
export function login(email: string, password: string): Promise<AuthResponse> {
  return apiFetch<AuthResponse>("/v1/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
}

export function signup(email: string, password: string): Promise<AuthResponse> {
  return apiFetch<AuthResponse>("/v1/auth/signup", { method: "POST", body: JSON.stringify({ email, password }) });
}

export function me(): Promise<User> {
  return apiFetch<User>("/v1/auth/me");
}

// Re-issue a fresh token for the current (still-valid) session — a sliding session.
export function refresh(): Promise<AuthResponse> {
  return apiFetch<AuthResponse>("/v1/auth/refresh", { method: "POST" });
}

// --- Chats ---
export function listChats(): Promise<ChatSession[]> {
  return apiFetch<ChatSession[]>("/v1/chats");
}

export function createChat(title?: string): Promise<ChatSession> {
  return apiFetch<ChatSession>("/v1/chats", { method: "POST", body: JSON.stringify(title ? { title } : {}) });
}

export function getChat(chatId: string): Promise<ChatDetail> {
  return apiFetch<ChatDetail>(`/v1/chats/${chatId}`);
}

export function renameChat(chatId: string, title: string): Promise<ChatSession> {
  return apiFetch<ChatSession>(`/v1/chats/${chatId}`, { method: "PATCH", body: JSON.stringify({ title }) });
}

export function deleteChat(chatId: string): Promise<ChatSession> {
  return apiFetch<ChatSession>(`/v1/chats/${chatId}`, { method: "DELETE" });
}

// Non-streaming fallback (the UI uses the SSE path in lib/sse.ts for live answers).
export function sendMessage(chatId: string, query: string): Promise<Message> {
  return apiFetch<Message>(`/v1/chats/${chatId}/messages`, { method: "POST", body: JSON.stringify({ query }) });
}

// --- Documents (knowledge base) ---
export function listDocuments(): Promise<DocumentItem[]> {
  return apiFetch<DocumentItem[]>("/v1/documents");
}

export function uploadDocument(file: File): Promise<IngestionJob> {
  const form = new FormData();
  form.append("file", file);
  // apiFetch leaves Content-Type unset for FormData so the browser sets the multipart boundary.
  return apiFetch<IngestionJob>("/v1/documents/upload", { method: "POST", body: form });
}

export function getIngestionJob(jobId: string): Promise<IngestionJob> {
  return apiFetch<IngestionJob>(`/v1/ingestion-jobs/${jobId}`);
}

export function deleteDocument(documentId: string): Promise<DocumentItem> {
  return apiFetch<DocumentItem>(`/v1/documents/${documentId}`, { method: "DELETE" });
}
