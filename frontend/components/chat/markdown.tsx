"use client";

import type { Element, Root, Text } from "hast";
import { CheckIcon, CopyIcon } from "lucide-react";
import { useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";

import { CitationChip } from "@/components/chat/citation-chip";
import { Button } from "@/components/ui/button";
import type { Citation } from "@/lib/types";

const CITATION_RE = /\[(\d+)\]/g;

// Turn inline [n] markers (outside code) into <citation> elements the renderer maps to chips.
// Runs before rehype-highlight so code blocks (parent <code>/<pre>) are skipped untouched.
function rehypeCitations() {
  return (tree: Root) => {
    visit(tree, "text", (node: Text, index, parent) => {
      if (index === undefined || !parent) return;
      if (parent.type === "element" && (parent.tagName === "code" || parent.tagName === "pre")) {
        return;
      }
      CITATION_RE.lastIndex = 0;
      if (!CITATION_RE.test(node.value)) return;

      CITATION_RE.lastIndex = 0;
      const replacement: Array<Text | Element> = [];
      let last = 0;
      let match: RegExpExecArray | null;
      while ((match = CITATION_RE.exec(node.value)) !== null) {
        if (match.index > last) {
          replacement.push({ type: "text", value: node.value.slice(last, match.index) });
        }
        replacement.push({
          type: "element",
          tagName: "citation",
          properties: { dataCitation: match[1] },
          children: [{ type: "text", value: match[0] }],
        });
        last = match.index + match[0].length;
      }
      if (last < node.value.length) {
        replacement.push({ type: "text", value: node.value.slice(last) });
      }
      (parent.children as Array<Text | Element>).splice(index, 1, ...replacement);
      return index + replacement.length;
    });
  };
}

function CodeBlock({ children }: { children?: React.ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(ref.current?.textContent ?? "");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — ignore
    }
  }

  return (
    <div className="group/code relative my-3">
      <pre ref={ref} className="overflow-x-auto rounded-lg border bg-background p-3 text-xs leading-relaxed">
        {children}
      </pre>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label="Copy code"
        onClick={copy}
        className="absolute top-2 right-2 opacity-0 transition-opacity group-hover/code:opacity-100"
      >
        {copied ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
      </Button>
    </div>
  );
}

const MARKDOWN_CLASSES =
  "text-sm leading-relaxed break-words [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 " +
  "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 " +
  "[&_h1]:mt-3 [&_h1]:mb-1 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mt-3 [&_h2]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold " +
  "[&_h3]:mt-2 [&_h3]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold " +
  "[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground " +
  "[&_table]:my-2 [&_table]:block [&_table]:overflow-x-auto [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 " +
  "[&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1";

export function Markdown({ content, citations }: { content: string; citations: Citation[] }) {
  const components = {
    pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
    code: ({ className, children }) => {
      const isBlock = /language-|hljs/.test(className ?? "");
      if (isBlock) {
        return <code className={className}>{children}</code>;
      }
      return (
        <code className="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-[0.85em]">
          {children}
        </code>
      );
    },
    a: ({ href, children }) => (
      <a href={href} className="font-medium underline underline-offset-2" target="_blank" rel="noreferrer">
        {children}
      </a>
    ),
    citation: ({ node }: { node?: Element }) => {
      const n = Number(node?.properties?.dataCitation);
      return <CitationChip n={n} citation={Number.isFinite(n) ? citations[n - 1] : undefined} />;
    },
  } as Components;

  return (
    <div className={MARKDOWN_CLASSES}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeCitations, rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
