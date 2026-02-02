'use client';

import Link from "next/link";

const REPO_URL = "https://github.com/lakshitsachdeva/intent2model";

export default function DriftPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col relative overflow-hidden font-mono">
      {/* Static gradient background — no WebGL, no lag */}
      <div className="absolute inset-0 z-0 bg-gradient-to-br from-black via-neutral-950 to-black" />

      {/* Nav */}
      <nav className="border-b border-white/10 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 sm:h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center">
            <span className="font-bold text-xl sm:text-2xl tracking-tight text-white">drift</span>
          </Link>
          <div className="flex items-center gap-3 sm:gap-4">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-white/80 hover:text-white transition-colors flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
              GitHub
            </a>
            <Link href="/" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Home
            </Link>
            <Link href="/drift#setup" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Setup
            </Link>
            <Link href="/drift#library" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Library
            </Link>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 max-w-4xl mx-auto py-10 sm:py-16 px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            ✓ Works on any laptop
          </span>
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
            macOS · Linux · Windows
          </span>
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30">
            Fully open source
          </span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold text-white mb-3 sm:mb-4">drift</h1>
        <p className="text-lg sm:text-xl text-white/70 mb-8 sm:mb-12">
          Terminal-first AutoML agent. Zero friction. Same engine as the web app.
        </p>

        {/* Repo CTA */}
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-4 py-3 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 transition-all mb-12 sm:mb-16"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
          <span className="font-medium text-white">github.com/lakshitsachdeva/intent2model</span>
          <span className="text-white/50 text-sm">→</span>
        </a>

        <section id="setup" className="space-y-6 sm:space-y-8 mb-10 sm:mb-12 scroll-mt-20">
          <h2 className="text-xl sm:text-2xl font-semibold text-white">Step-by-step setup</h2>
          <p className="text-white/70">
            drift needs <strong>both pipx and npm</strong>. pipx runs the Python CLI (the chat REPL). npm starts the engine and downloads it. Follow in order:
          </p>

          {/* Step 1: LLM */}
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-500/30 text-emerald-400 font-bold text-sm">1</span>
              <h3 className="text-white font-semibold text-lg">Get a local LLM</h3>
            </div>
            <p className="text-white/70 text-sm">
              drift uses an LLM for planning and training. Pick one <strong>before</strong> installing drift. See options below (Gemini CLI or Ollama).
            </p>
          </div>

          {/* Step 2: pipx */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-500/30 text-blue-400 font-bold text-sm">2</span>
              <h3 className="text-white font-semibold text-lg">Install pipx</h3>
            </div>
            <p className="text-white/70 text-sm">
              pipx runs the Python CLI (the actual drift chat interface). Without it, drift won&apos;t start.
            </p>
            <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`pip install pipx
pipx ensurepath`}
            </pre>
            <p className="text-white/50 text-xs">Windows: restart PowerShell after ensurepath. macOS/Linux: restart terminal or run <code className="bg-white/10 px-1 rounded">source ~/.zshrc</code></p>
          </div>

          {/* Step 3: npm */}
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/30 text-purple-400 font-bold text-sm">3</span>
              <h3 className="text-white font-semibold text-lg">Install npm (Node.js)</h3>
            </div>
            <p className="text-white/70 text-sm">
              npm starts the engine, downloads it on first run, and launches the pipx CLI. Get it from <a href="https://nodejs.org" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">nodejs.org</a>.
            </p>
          </div>

          {/* Step 4: drift */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-amber-500/30 text-amber-400 font-bold text-sm">4</span>
              <h3 className="text-white font-semibold text-lg">Install drift (both)</h3>
            </div>
            <p className="text-white/70 text-sm">
              You need <strong>both</strong> — npm for the engine launcher, pipx for the CLI.
            </p>
            <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`pipx install drift-ml
npm install -g drift-ml`}
            </pre>
          </div>

          {/* Step 5: Run */}
          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-cyan-500/30 text-cyan-400 font-bold text-sm">5</span>
              <h3 className="text-white font-semibold text-lg">Run drift</h3>
            </div>
            <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
drift
            </pre>
            <p className="text-white/50 text-sm">First run downloads the engine (~100MB). Then:</p>
            <pre className="bg-black/60 border border-white/20 text-cyan-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`drift › load data.csv
drift › predict price
drift › try something stronger
drift › quit`}
            </pre>
          </div>

          <section id="library" className="scroll-mt-20 pt-8 space-y-4">
            <h2 className="text-xl sm:text-2xl font-semibold text-white">Use as library</h2>
            <p className="text-white/70">
              Add drift to your Python scripts. <code className="bg-white/10 px-1.5 py-0.5 rounded">pip install drift-ml</code> and import.
            </p>
            <pre className="bg-black/60 border border-white/20 text-green-400 p-3 sm:p-4 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`from drift import Drift

d = Drift()
d.load("iris.csv")
d.chat("predict sepal length")
result = d.train()
print(result["metrics"])`}
            </pre>
            <p className="text-white/50 text-sm">
              Or connect to an existing engine: <code className="bg-white/10 px-1 rounded">Drift(base_url="http://localhost:8000")</code>
            </p>
          </section>

          <h2 className="text-xl sm:text-2xl font-semibold text-white pt-8">LLM options — pick one</h2>
          <p className="text-white/70">
            Training and planning use an LLM. You need one. Here&apos;s exactly how to install each option:
          </p>

          {/* Gemini CLI */}
          <div className="bg-black/40 border border-white/20 rounded-xl p-6 sm:p-8 space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="text-white font-semibold text-lg">Option A: Gemini CLI</h3>
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400">Recommended</span>
            </div>
            <p className="text-white/60 text-sm">
              Google&apos;s Gemini in your terminal. Free tier. drift uses it by default if <code className="bg-white/10 px-1 rounded">gemini</code> is on PATH.
            </p>
            <div className="space-y-3">
              <p className="text-white/50 text-xs font-medium uppercase tracking-wider">Install Gemini CLI</p>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">npm (any platform)</p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
npm install -g @google/gemini-cli
                </pre>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Homebrew (macOS / Linux)</p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
brew install gemini-cli
                </pre>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Get your API key</p>
                <p className="text-white/50 text-sm mt-1">
                  Go to <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">aistudio.google.com/apikey</a>, create a key, then:
                </p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
export GEMINI_API_KEY=&quot;your-key-here&quot;
                </pre>
                <p className="text-white/50 text-xs mt-1">Add to <code className="bg-white/10 px-1 rounded">~/.zshrc</code> or <code className="bg-white/10 px-1 rounded">~/.bashrc</code> to persist.</p>
              </div>
            </div>
          </div>

          {/* Ollama / Llama */}
          <div className="bg-black/40 border border-white/20 rounded-xl p-6 sm:p-8 space-y-4">
            <h3 className="text-white font-semibold text-lg">Option B: Ollama + Llama</h3>
            <p className="text-white/60 text-sm">
              Run Llama (and other models) locally. No API key. Fully offline.
            </p>
            <div className="space-y-3">
              <p className="text-white/50 text-xs font-medium uppercase tracking-wider">Install Ollama</p>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">macOS</p>
                <p className="text-white/50 text-sm">Download from <a href="https://ollama.com/download/mac" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">ollama.com/download/mac</a> — drag to Applications.</p>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Linux</p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
curl -fsSL https://ollama.com/install.sh | sh
                </pre>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Windows</p>
                <p className="text-white/50 text-sm">Download from <a href="https://ollama.com/download/windows" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">ollama.com/download/windows</a> — run the installer.</p>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Download Llama (after Ollama is installed)</p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
ollama pull llama3.2
                </pre>
                <p className="text-white/50 text-xs mt-1">Or <code className="bg-white/10 px-1 rounded">ollama pull llama2</code>, <code className="bg-white/10 px-1 rounded">ollama pull gemma2</code>, etc.</p>
              </div>
              <div className="space-y-2">
                <p className="text-white/40 text-xs">Run the model (keep this running in another terminal)</p>
                <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
ollama run llama3.2
                </pre>
                <p className="text-white/50 text-xs mt-1">Or run once in background: <code className="bg-white/10 px-1 rounded">ollama serve</code></p>
              </div>
            </div>
          </div>

          {/* Other */}
          <div className="bg-black/40 border border-white/20 rounded-xl p-6 sm:p-8 space-y-4">
            <h3 className="text-white font-semibold text-lg">Option C: Other local LLM</h3>
            <p className="text-white/60 text-sm">
              Any compatible API. Configure the engine accordingly. See the <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">repo</a> for details.
            </p>
          </div>

          <p className="text-white/50 text-sm">
            The engine runs planning and training locally. The LLM is the brain; execution is automatic. No data leaves your machine.
          </p>

          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 space-y-3">
            <h3 className="text-amber-400 font-semibold">Windows: engine crashes?</h3>
            <p className="text-white/70 text-sm">
              Run the engine manually in PowerShell to see the error:
            </p>
            <pre className="bg-black/60 border border-white/20 text-green-400 p-3 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`cd $env:USERPROFILE\\.drift\\bin
.\\drift-engine-windows-x64.exe`}
            </pre>
            <p className="text-white/60 text-sm">
              Fixes: <a href="https://aka.ms/vs/17/release/vc_redist.x64.exe" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">Visual C++ Redistributable</a>, Windows Defender exception, allow port 8000 in firewall.
            </p>
          </div>
        </section>

        <section id="run-locally" className="space-y-4 sm:space-y-6 mb-10 sm:mb-12 scroll-mt-20">
          <h2 className="text-xl sm:text-2xl font-semibold text-white">Run locally</h2>
          <p className="text-white/70 text-sm">
            Clone the repo and start the engine:
          </p>
          <pre className="bg-black/60 border border-white/20 text-green-400 p-3 sm:p-4 rounded-lg overflow-x-auto text-xs sm:text-sm">
{`git clone ${REPO_URL}.git
cd intent2model
./start.sh`}
          </pre>
          <p className="text-white/50 text-sm">
            Deploy the frontend to Vercel so anyone can open the UI. The engine runs on each user&apos;s machine (or set <code className="bg-white/10 px-2 py-1 rounded text-green-400">DRIFT_BACKEND_URL</code> to a running engine).
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl sm:text-2xl font-semibold text-white">What is drift?</h2>
          <p className="text-white/70">
            Local-first: the engine runs on your machine. Training and planning stay local; you never send data to our servers. Terminal-first, chat-based — same engine as the web app. No commands to memorize. Zero auth. Zero tokens. Fully open source.
          </p>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            github.com/lakshitsachdeva/intent2model
          </a>
        </section>
      </main>

      <footer className="border-t border-white/10 py-6 sm:py-8 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 text-center space-y-2">
          <p className="text-sm text-white/50">
            by Lakshit Sachdeva &copy; 2026
          </p>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="text-cyan-500/70 hover:text-cyan-400 text-sm transition-colors">
            Star on GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}
