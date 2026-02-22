import Link from "next/link";
import { Terminal } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-border bg-card/50 mt-auto">
      <div className="max-w-[1440px] mx-auto px-6 md:px-8 py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 items-center">
          
          {/* Left: Branding & Version */}
          <div className="flex flex-row items-center gap-3 sm:gap-4 text-muted-foreground group">
            <div className="flex items-center gap-2">
              <Terminal className="size-4 group-hover:text-primary transition-colors" />
              <span className="font-display text-sm tracking-widest text-foreground">MATCHA AI</span>
            </div>
            <div className="w-px h-3 bg-border" />
            <span className="font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.1em]">v2.1.0-RC</span>
          </div>

          {/* Center: System Status - Hidden on extra small */}
          <div className="hidden sm:flex items-center justify-start lg:justify-center gap-6">
            <div className="flex items-center gap-2">
              <span className="size-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
              <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">ORCHESTRATOR OK</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
              <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">INFERENCE READY</span>
            </div>
          </div>

          {/* Right: Links */}
          <div className="flex items-center justify-start sm:justify-end lg:justify-end gap-6 sm:gap-8 font-mono text-[10px] uppercase tracking-widest sm:col-span-2 lg:col-span-1 border-t sm:border-t-0 border-border pt-4 sm:pt-0">
            <Link href="/" className="text-muted-foreground hover:text-primary transition-colors">
              DASHBOARD
            </Link>
            <a href="https://github.com/matcha-ai" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary transition-colors">
              SOURCE
            </a>
          </div>

        </div>
        
        {/* Bottom Rule equivalent */}
        <div className="mt-6 pt-4 border-t border-border flex justify-between items-center text-[9px] font-mono text-muted-foreground/60 uppercase tracking-widest">
          <span>SECURE UPLINK ESTABLISHED</span>
          <span>© {new Date().getFullYear()} MATCHA RESEARCH</span>
        </div>
      </div>
    </footer>
  );
}
