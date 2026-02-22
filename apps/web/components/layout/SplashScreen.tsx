"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";

export function SplashScreen() {
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Simulate initial system load time (e.g., fetching configs, establishing connections)
    const totalDuration = 2000; // 2 seconds
    const interval = 50; // Update every 50ms
    const step = (100 / (totalDuration / interval));

    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(() => setLoading(false), 400); // Small buffer at 100% before fading out
          return 100;
        }
        // Add a slight randomization to make it feel like real loading
        return prev + step * (Math.random() * 1.5 + 0.5); 
      });
    }, interval);

    return () => clearInterval(timer);
  }, []);

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.05, filter: "brightness(2)" }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background pointer-events-auto"
        >
          {/* Subtle background noise for the splash screen */}
          <div className="absolute inset-0 z-0 h-full w-full opacity-[0.03] mix-blend-overlay bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />
          
          <div className="relative z-10 flex flex-col items-center">
            {/* Blinking Logo */}
            <motion.div 
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
              className="relative size-24 md:size-32 mb-8 drop-shadow-[0_0_24px_rgba(var(--color-primary),0.4)]"
            >
              <Image 
                src="/favicons/logo.png" 
                alt="Matcha AI Loading" 
                fill
                className="object-contain"
                priority
                sizes="(max-width: 768px) 96px, 128px"
              />
            </motion.div>

            {/* Core Systems Boot text */}
            <div className="font-mono text-[10px] md:text-xs text-muted-foreground uppercase tracking-[0.3em] mb-4 animate-pulse">
              Initializing Core Systems...
            </div>

            {/* Progress Bar Track */}
            <div className="w-64 md:w-80 h-1 bg-muted/30 overflow-hidden rounded-full border border-border/50">
              {/* Progress Bar Fill */}
              <div 
                className="h-full bg-primary shadow-[0_0_10px_rgba(var(--color-primary),1)] transition-all duration-75 ease-out"
                style={{ width: `${Math.min(progress, 100)}%` }}
              />
            </div>
            
            {/* Percentage Text */}
            <div className="mt-3 font-mono text-[9px] text-primary/80 tracking-widest">
              {Math.min(100, Math.floor(progress)).toString().padStart(3, '0')}%
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
