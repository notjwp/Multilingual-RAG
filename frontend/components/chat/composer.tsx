"use client";

import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ComposerProps {
  onSend: (query: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
}

export function Composer({ onSend, onStop, streaming, disabled }: ComposerProps) {
  const [value, setValue] = useState("");

  function submit() {
    const query = value.trim();
    if (!query || streaming || disabled) return;
    onSend(query);
    setValue("");
  }

  return (
    <form
      className="relative"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <Textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        disabled={disabled}
        rows={1}
        placeholder="Message… (Enter to send, Shift+Enter for a new line)"
        className="max-h-48 min-h-12 resize-none pr-12"
      />
      <div className="absolute right-2 bottom-2">
        {streaming ? (
          <Button
            type="button"
            size="icon-sm"
            variant="secondary"
            onClick={onStop}
            aria-label="Stop generating"
          >
            <SquareIcon className="size-4" />
          </Button>
        ) : (
          <Button
            type="submit"
            size="icon-sm"
            disabled={!value.trim() || disabled}
            aria-label="Send message"
          >
            <ArrowUpIcon className="size-4" />
          </Button>
        )}
      </div>
    </form>
  );
}
