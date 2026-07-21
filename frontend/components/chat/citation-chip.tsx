"use client";

import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { Citation } from "@/lib/types";

function sourceLabel(citation: Citation): string {
  // `source` is a stored path; show just the file name.
  const name = citation.source.split(/[\\/]/).pop() || citation.source;
  return citation.page != null ? `${name} · p.${citation.page}` : name;
}

// Inline [n] marker → a clickable chip whose popover shows the cited chunk.
export function CitationChip({ n, citation }: { n: number; citation?: Citation }) {
  if (!citation) {
    // Marker with no matching citation (e.g. still streaming) — render as plain text.
    return <sup className="text-muted-foreground">[{n}]</sup>;
  }
  return (
    <Popover>
      <PopoverTrigger
        render={
          <button
            type="button"
            className="mx-0.5 inline-flex items-center rounded bg-primary/10 px-1 align-super text-[0.7em] font-medium text-primary transition-colors hover:bg-primary/20"
          />
        }
      >
        {n}
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <PopoverHeader>
          <PopoverTitle className="text-xs break-words">{sourceLabel(citation)}</PopoverTitle>
        </PopoverHeader>
        <PopoverDescription className="max-h-48 overflow-y-auto text-xs leading-relaxed whitespace-pre-wrap">
          {citation.text}
        </PopoverDescription>
      </PopoverContent>
    </Popover>
  );
}

// The "Sources" block rendered beneath a grounded answer.
export function SourcesList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <div className="mt-3 border-t pt-2">
      <p className="mb-1.5 text-xs font-medium text-muted-foreground">Sources</p>
      <ol className="flex flex-col gap-1">
        {citations.map((citation, i) => (
          <li key={citation.chunk_id} className="flex gap-1.5 text-xs">
            <span className="shrink-0 font-medium text-primary">[{i + 1}]</span>
            <span className="min-w-0 truncate text-muted-foreground" title={citation.text}>
              {sourceLabel(citation)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
