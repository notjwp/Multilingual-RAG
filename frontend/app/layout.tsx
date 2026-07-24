import type { Metadata } from "next";
import { Geist_Mono, Lato, Poppins } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

// Poppins for headings, Lato for body — a modern, highly readable pairing. Self-hosted by
// next/font (no external requests, CSP-safe). Geist Mono stays for code.
const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const lato = Lato({
  variable: "--font-lato",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Multilingual RAG",
  description: "A multilingual retrieval-augmented chat assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${poppins.variable} ${lato.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
