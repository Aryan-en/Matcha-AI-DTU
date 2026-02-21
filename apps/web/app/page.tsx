import { VideoUpload } from "@/components/video-upload";
import { MatchDashboard } from "@/components/match-dashboard";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0a0a0a]">
      {/* Nav */}
      <nav className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between bg-[#0f0f0f]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-emerald-500 flex items-center justify-center font-bold text-black text-lg font-heading">M</div>
          <span className="font-heading font-semibold text-zinc-100 text-lg tracking-tight">MATCHA <span className="text-emerald-400">AI</span></span>
        </div>
        <span className="text-xs text-zinc-400 border border-zinc-700 px-4 py-1.5 font-mono uppercase tracking-wider bg-zinc-900">DTU Edition</span>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-2 gap-16 items-start">
        {/* Left: Hero + Upload */}
        <div className="space-y-10">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 text-xs font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-4 py-2 uppercase tracking-wider">
              <span className="w-2 h-2 bg-emerald-400 animate-pulse" />
              AI Pipeline Active
            </div>
            <h1 className="font-heading text-5xl lg:text-6xl font-bold tracking-tight leading-[1.1] text-white">
              Automated <span className="text-emerald-400">Sports</span><br />
              Highlights &amp; Commentary
            </h1>
            <p className="text-zinc-400 text-lg leading-relaxed max-w-lg">
              Upload match footage. Our AI detects goals, fouls, saves &amp; tackles, scores each moment, and builds highlight reels with live commentary.
            </p>
          </div>

          {/* Feature pills */}
          <div className="flex flex-wrap gap-3">
            {["Event Detection", "Motion Analysis", "Context Scoring", "Highlight Selection", "Auto Commentary", "Timeline View"].map((f) => (
              <span key={f} className="text-xs text-zinc-400 bg-zinc-900 border border-zinc-700 px-4 py-2 hover:border-emerald-500 hover:text-emerald-400 transition-colors uppercase tracking-wide font-medium">{f}</span>
            ))}
          </div>

          {/* Upload card */}
          <div className="bg-[#141414] border border-zinc-800 p-6">
            <h2 className="font-heading text-xl font-semibold mb-5 text-zinc-100 tracking-tight">Upload Match Footage</h2>
            <VideoUpload />
          </div>
        </div>

        {/* Right: Dashboard */}
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="font-heading text-xl font-semibold text-zinc-100 tracking-tight">Recent Matches</h2>
          </div>
          <MatchDashboard />
        </div>
      </div>
    </main>
  );
}

