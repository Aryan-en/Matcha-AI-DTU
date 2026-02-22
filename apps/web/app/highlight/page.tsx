"use client";

import { useEffect } from "react";
import { redirect } from "next/navigation";

/**
 * Legacy /highlight route shim.
 * Redirects to dashboard because highlights live under /matches/[id]#highlights.
 */
export default function HighlightRedirect() {
  useEffect(() => {
    redirect("/");
  }, []);

  return null;
}
