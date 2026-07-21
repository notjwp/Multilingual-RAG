"use client";

import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type Mode = "login" | "signup";

const COPY: Record<Mode, {
  title: string;
  desc: string;
  cta: string;
  alt: string;
  altHref: string;
  altCta: string;
}> = {
  login: {
    title: "Welcome back",
    desc: "Sign in to continue.",
    cta: "Sign in",
    alt: "Need an account?",
    altHref: "/signup",
    altCta: "Sign up",
  },
  signup: {
    title: "Create your account",
    desc: "Start chatting with your documents.",
    cta: "Create account",
    alt: "Already have an account?",
    altHref: "/login",
    altCta: "Sign in",
  },
};

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function AuthForm({ mode }: { mode: Mode }) {
  const auth = useAuth();
  const router = useRouter();
  const reduce = useReducedMotion();
  const copy = COPY[mode];

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.SyntheticEvent) {
    e.preventDefault();
    if (!EMAIL_RE.test(email)) {
      setError("Enter a valid email address.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") await auth.login(email, password);
      else await auth.signup(email, password);
      router.replace("/");
    } catch (err) {
      setError(errorMessage(err));
      setSubmitting(false);
    }
  }

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{copy.title}</CardTitle>
          <CardDescription>{copy.desc}</CardDescription>
        </CardHeader>
        <form onSubmit={onSubmit} noValidate>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                placeholder="At least 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <Button type="submit" className="mt-1 w-full" disabled={submitting}>
              {submitting ? "Please wait…" : copy.cta}
            </Button>
          </CardContent>
        </form>
        <CardFooter className="justify-center">
          <p className="text-sm text-muted-foreground">
            {copy.alt}{" "}
            <Link
              href={copy.altHref}
              className="font-medium text-foreground underline-offset-4 hover:underline"
            >
              {copy.altCta}
            </Link>
          </p>
        </CardFooter>
      </Card>
    </motion.div>
  );
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.code) {
      case "invalid_credentials":
        return "Incorrect email or password.";
      case "email_already_registered":
        return "That email is already registered. Try signing in instead.";
      case "validation_error":
        return "Please check your email and password and try again.";
      default:
        return err.message;
    }
  }
  return "Something went wrong. Please try again.";
}
