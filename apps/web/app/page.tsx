import { VideoUpload } from "@/components/video-upload";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Matcha AI &nbsp;
          <code className="font-mono font-bold">dtu-edition</code>
        </p>
      </div>

      <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:rounded-full before:bg-gradient-to-br before:from-emerald-400 before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-to-br after:from-sky-200 after:via-sky-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-emerald-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px] z-[-1]">
        <div className="flex flex-col items-center gap-8 text-center">
          <h1 className="text-5xl font-bold tracking-tight text-slate-900 dark:text-slate-100 sm:text-7xl">
            Automated <span className="text-emerald-500">Sports</span> <br />
            Commentary & Highlights
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl">
            Upload your match footage and let our AI generate live commentary,
            detect key events, and create shareable highlights in minutes.
          </p>
        </div>
      </div>

      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:text-left z-10 mt-12">
        <div className="bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-3xl shadow-xl border border-slate-200 dark:border-slate-800 p-8">
            <h2 className="text-2xl font-semibold mb-6 text-center text-slate-800 dark:text-slate-200">
                Get Started
            </h2>
            <VideoUpload />
        </div>
      </div>
    </main>
  );
}
