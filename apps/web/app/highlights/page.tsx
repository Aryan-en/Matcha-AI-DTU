"use client";

import { useEffect } from "react";
import { redirect } from "next/navigation";

/**
 * Legacy /highlights route shim.
 * Redirects to dashboard because highlights live under /matches/[id]#highlights.
 */
export default function HighlightsRedirect() {
  useEffect(() => {
    redirect("/");
  }, []);

  return null;
}
