// TypeScript mirror of the FastAPI backend's request/response models.
// Field names are verified against the live API contract (see the M16 plan).

export interface User {
  user_id: string;
  email: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ChatSession {
  session_id: string;
  title: string;
  created_at: string;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  source: string;
  page: number | null;
  text: string;
}

export type Role = "user" | "assistant";

export interface Message {
  message_id: string;
  role: Role;
  content: string;
  created_at: string;
  citations: Citation[];
}

export interface ChatDetail {
  session: ChatSession;
  messages: Message[];
}

// Part 5 (deferred) — documents & ingestion. Kept here so the client stays single-sourced.
export type IngestionStatus = "queued" | "running" | "succeeded" | "failed";

export interface IngestionJob {
  job_id: string;
  status: IngestionStatus;
  document_id?: string | null;
  error_message?: string | null;
}

export interface DocumentMeta {
  document_id: string;
  source: string;
  content_type: string;
  language: string;
  checksum?: string;
  created_at?: string;
}

// A row from GET /v1/chats/{chatId}/documents. Note: ingestion status lives on the job, not here.
export interface DocumentItem {
  document: DocumentMeta;
  chunk_count: number;
}

export interface ApiErrorBody {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}
