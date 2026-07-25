# Frontend — Multilingual RAG

The chat UI: multi-session chat with persisted history, token-by-token streaming answers,
markdown + inline citations, and per-chat document uploads.

Next.js 16 (App Router) · React 19 · Tailwind v4 · [Base UI](https://base-ui.com) (shadcn
`base-nova` style, **not** Radix) · Poppins headings / Lato body.

## Run it

The backend must be running first (API on `:8000`, plus Postgres, Redis, and the Celery worker —
see the root README). Then:

```bash
npm install
npm run dev            # http://localhost:3000
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` if the API isn't at `http://localhost:8000`. It is
baked into the client bundle at build time, so it must be the URL the **browser** uses.

Or run the whole stack (Postgres · Redis · API · worker · frontend) with one command from the repo
root: `docker compose up --build`.

## Verify

```bash
npm run lint
npm run build
```

Both run in CI on every push.

## Layout

```text
app/(app)/          authenticated shell — chat routes
app/login, /signup  auth pages
components/chat/    chat window, composer (+ paperclip upload), sidebar, message bubbles, files dialog
components/ui/      Base UI primitives (button, dialog, sheet, …)
lib/api.ts          typed API client — the one place that talks to the backend
lib/sse.ts          SSE consumer (fetch + ReadableStream; EventSource can't send auth headers)
lib/auth.tsx        auth context, token storage, background refresh
lib/chats.tsx       chat list state
```

## Gotchas

- **Auth is a Bearer token in `localStorage`** (`mrag_token`), refreshed in the background via
  `/v1/auth/refresh`. `lib/api.ts` is the only module that touches it.
- **Streaming can't use `EventSource`** — it can't POST or set an `Authorization` header, so
  `lib/sse.ts` reads the response stream manually and parses SSE frames.
- **Documents are per chat.** Upload from the paperclip in the composer; a file grounds only that
  chat. There is no global documents page.
- **Base UI, not Radix.** Custom triggers use `render={<Button/>}`; dialogs are controlled via
  `open` / `onOpenChange`.
- After deleting or moving a route, a stale `.next/dev/types` validator can fail the build —
  `rm -rf .next` and rebuild.
- See `AGENTS.md`: this Next.js version has breaking changes; check `node_modules/next/dist/docs/`
  before relying on older Next patterns.
