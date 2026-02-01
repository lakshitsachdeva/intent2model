import Link from "next/link";

export default function DriftPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex flex-col">
      <nav className="border-b bg-white dark:bg-black/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold">i2</span>
            </div>
            <span className="font-bold text-xl tracking-tight">Intent2Model</span>
          </Link>
          <div className="flex items-center space-x-4">
            <Link href="/" className="text-sm font-medium hover:text-primary transition-colors">
              Web app
            </Link>
            <Link href="/drift#setup" className="text-sm font-medium hover:text-primary transition-colors">
              Docs &amp; setup
            </Link>
            <Link href="/drift#run-locally" className="px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-full text-sm font-medium hover:opacity-90 transition-opacity">
              Run locally
            </Link>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-3xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">drift</h1>
        <p className="text-muted-foreground mb-8">
          Terminal-first AutoML agent. Same engine as the web UI — chat-based, no commands to memorize.
        </p>

        <p className="text-muted-foreground mb-8">
          Local-first: training and planning run on your machine. The web app can be hosted on Vercel so anyone can open the UI; the engine runs locally (or you point to your own). You need an LLM: Gemini CLI, Ollama/Llama, or another local LLM — see Setup below.
        </p>

        <section className="space-y-4 mb-10">
          <h2 className="text-xl font-semibold">What is drift?</h2>
          <p className="text-muted-foreground">
            drift is a chat-based CLI that uses the same engine as this web app. You talk in natural language: load a dataset, ask to predict a column, try stronger models, or ask why accuracy is capped. The engine runs locally; no commands to memorize.
          </p>
        </section>

        <section id="setup" className="space-y-4 mb-10 scroll-mt-8">
          <h2 className="text-xl font-semibold">Docs &amp; setup — LLM required</h2>
          <p className="text-muted-foreground">
            Training and planning use an LLM. You need one of these (or a Gemini API key for cloud):
          </p>
          <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm">
            <li><strong className="text-foreground">Gemini CLI</strong> — Install the Google Gemini CLI, set <code className="bg-muted px-1 rounded">GEMINI_API_KEY</code> or use <code className="bg-muted px-1 rounded">gemini</code> on your PATH. The engine uses it by default.</li>
            <li><strong className="text-foreground">Ollama / Llama</strong> — Run Ollama locally (e.g. <code className="bg-muted px-1 rounded">ollama run llama2</code>). Point the engine to your Ollama URL if needed.</li>
            <li><strong className="text-foreground">Other local LLM</strong> — Any compatible API. Configure the engine accordingly.</li>
          </ul>
          <p className="text-muted-foreground text-sm">
            The web app and drift CLI both talk to the same engine. The engine runs planning and training locally; the LLM is the brain, execution is automatic.
          </p>
        </section>

        <section id="run-locally" className="space-y-4 mb-10 scroll-mt-8">
          <h2 className="text-xl font-semibold">Run locally</h2>
          <p className="text-muted-foreground text-sm mb-4">From the project root:</p>
          <pre className="bg-zinc-900 dark:bg-zinc-800 text-zinc-100 p-4 rounded-lg overflow-x-auto text-sm font-mono">
            ./start.sh
          </pre>
          <p className="text-muted-foreground text-sm">
            Deploy the frontend to Vercel so anyone can open the UI. The engine runs on each user’s machine (or they set <code className="bg-muted px-1 rounded">DRIFT_BACKEND_URL</code> to a running engine).
          </p>
        </section>

        <section className="space-y-4 mb-10">
          <h2 className="text-xl font-semibold">Install drift CLI</h2>
          <p className="text-muted-foreground text-sm mb-4">npm (global launcher; requires Python):</p>
          <pre className="bg-zinc-900 dark:bg-zinc-800 text-zinc-100 p-4 rounded-lg overflow-x-auto text-sm font-mono">
            npm install -g drift-ml{"\n"}
            drift
          </pre>
          <p className="text-muted-foreground text-sm mt-6 mb-4">pipx (Python-only):</p>
          <pre className="bg-zinc-900 dark:bg-zinc-800 text-zinc-100 p-4 rounded-lg overflow-x-auto text-sm font-mono">
            pipx install drift{"\n"}
            drift
          </pre>
          <p className="text-sm text-muted-foreground mt-4">
            The CLI starts the engine automatically on first run (or use <code className="bg-muted px-1 rounded">DRIFT_BACKEND_URL</code> to point to a running engine). You need an LLM (Gemini CLI, Ollama, etc.) as in Setup above.
          </p>
        </section>

        <section className="space-y-4 mb-10">
          <h2 className="text-xl font-semibold">Example usage</h2>
          <p className="text-muted-foreground text-sm mb-4">Start the CLI, then type naturally:</p>
          <pre className="bg-zinc-900 dark:bg-zinc-800 text-zinc-100 p-4 rounded-lg overflow-x-auto text-sm font-mono">
            drift › load iris.csv{"\n"}
            drift › predict sepal.length{"\n"}
            drift › try something stronger{"\n"}
            drift › why is accuracy capped
          </pre>
        </section>
      </main>

      <footer className="border-t py-8 bg-white dark:bg-black">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Built with ❤️ by lakshit for autonomous machine learning. &copy; 2026 Intent2Model Inc.
          </p>
        </div>
      </footer>
    </div>
  );
}
