import Link from "next/link";
import Prism from "@/components/Prism";

export default function DriftPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col relative overflow-hidden font-mono">
      {/* Prism background */}
      <div className="absolute inset-0 z-0 opacity-30">
        <Prism
          animationType="rotate"
          timeScale={0.5}
          height={3.5}
          baseWidth={5.5}
          scale={3.6}
          hueShift={0}
          colorFrequency={1}
          noise={0}
          glow={1}
        />
      </div>

      {/* Nav */}
      <nav className="border-b border-white/10 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center">
            <span className="font-bold text-2xl tracking-tight text-white">drift</span>
          </Link>
          <div className="flex items-center space-x-4">
            <Link href="/" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Home
            </Link>
            <Link href="/drift#setup" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Setup
            </Link>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 max-w-4xl mx-auto py-16 px-4 sm:px-6 lg:px-8 relative z-10">
        <h1 className="text-5xl font-bold text-white mb-4">drift</h1>
        <p className="text-xl text-white/70 mb-12">
          Terminal-first AutoML agent. Same engine as the web app.
        </p>

        <section className="space-y-6 mb-12">
          <h2 className="text-2xl font-semibold text-white">Exactly what to do</h2>
          <div className="space-y-4 text-white/80">
            <div>
              <p className="text-white/60 text-sm mb-2">1. Install drift</p>
              <pre className="bg-black/60 border border-white/20 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
npm install -g drift-ml
              </pre>
            </div>
            <div>
              <p className="text-white/60 text-sm mb-2">2. Run drift</p>
              <pre className="bg-black/60 border border-white/20 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
drift
              </pre>
              <p className="text-white/50 text-sm mt-2">
                On first run: downloads engine, starts it locally. You'll see the welcome in your terminal.
              </p>
            </div>
            <div>
              <p className="text-white/60 text-sm mb-2">3. Use a local LLM (required)</p>
              <ul className="list-disc list-inside text-white/60 text-sm space-y-1 ml-4">
                <li><span className="text-white">Gemini CLI</span> — install and set GEMINI_API_KEY</li>
                <li><span className="text-white">Ollama</span> — run ollama run llama2</li>
                <li><span className="text-white">Other local LLM</span> — configure engine accordingly</li>
              </ul>
            </div>
            <div>
              <p className="text-white/60 text-sm mb-2">4. Chat</p>
              <pre className="bg-black/60 border border-white/20 text-cyan-400 p-4 rounded-lg overflow-x-auto text-sm">
drift › load data.csv
drift › predict price
drift › try something stronger
drift › why is accuracy capped
drift › quit
              </pre>
            </div>
          </div>
        </section>

        <section id="setup" className="space-y-6 mb-12 scroll-mt-8">
          <h2 className="text-2xl font-semibold text-white">Setup — LLM required</h2>
          <p className="text-white/70">
            Training and planning use an LLM. You need one of these:
          </p>
          <div className="space-y-4 text-white/60 text-sm">
            <div className="bg-black/40 border border-white/20 p-4 rounded-lg">
              <h3 className="text-white font-semibold mb-2">Gemini CLI</h3>
              <p>Install Google Gemini CLI, set <code className="bg-white/10 px-2 py-1 rounded text-green-400">GEMINI_API_KEY</code> or have <code className="bg-white/10 px-2 py-1 rounded text-green-400">gemini</code> on PATH. Engine uses it by default.</p>
            </div>
            <div className="bg-black/40 border border-white/20 p-4 rounded-lg">
              <h3 className="text-white font-semibold mb-2">Ollama / Llama</h3>
              <p>Run Ollama locally: <code className="bg-white/10 px-2 py-1 rounded text-green-400">ollama run llama2</code>. Point engine to your Ollama URL if needed.</p>
            </div>
            <div className="bg-black/40 border border-white/20 p-4 rounded-lg">
              <h3 className="text-white font-semibold mb-2">Other local LLM</h3>
              <p>Any compatible API. Configure the engine accordingly.</p>
            </div>
          </div>
          <p className="text-white/50 text-sm">
            The engine runs planning and training locally. The LLM is the brain; execution is automatic.
          </p>
        </section>

        <section id="run-locally" className="space-y-6 mb-12 scroll-mt-8">
          <h2 className="text-2xl font-semibold text-white">Run locally</h2>
          <p className="text-white/70 text-sm">
            Clone the repo and start the engine:
          </p>
          <pre className="bg-black/60 border border-white/20 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
./start.sh
          </pre>
          <p className="text-white/50 text-sm">
            Deploy the frontend to Vercel so anyone can open the UI. The engine runs on each user's machine (or they set <code className="bg-white/10 px-2 py-1 rounded text-green-400">DRIFT_BACKEND_URL</code> to a running engine).
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="text-2xl font-semibold text-white">What is drift?</h2>
          <p className="text-white/70">
            Local-first: the engine runs on your machine. Training and planning stay local; you never send data to our servers. Terminal-first, chat-based — same engine as the web app. No commands to memorize.
          </p>
        </section>
      </main>

      <footer className="border-t border-white/10 py-8 relative z-10">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-white/50">
            by Lakshit Sachdeva &copy; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
