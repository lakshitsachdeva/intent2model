'use client';

import Link from "next/link";
import dynamic from "next/dynamic";
import TextType from "@/components/TextType";

const Prism = dynamic(() => import("@/components/Prism"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 bg-black" />,
});

const REPO_URL = "https://github.com/lakshitsachdeva/intent2model";

export default function Home() {
  return (
    <div className="min-h-screen bg-black flex flex-col relative overflow-hidden font-mono">
      {/* Prism background */}
      <div className="absolute inset-0 z-0">
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
              Star
            </a>
            <Link href="/drift" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              CLI
            </Link>
            <Link href="/drift#setup" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Docs
            </Link>
            <Link href="/drift#library" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Library
            </Link>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 py-8 sm:py-0 relative z-10">
        <div className="max-w-4xl w-full text-center">
          <h1 className="text-5xl sm:text-6xl md:text-8xl font-bold text-white mb-4 sm:mb-6">
            drift
          </h1>
          <div className="text-lg sm:text-2xl md:text-3xl text-white/90 mb-8 sm:mb-12 min-h-12 sm:h-12 flex items-center justify-center">
            <TextType
              texts={[
                "Terminal-first AutoML",
                "Chat-based ML engineer",
                "Local-first training",
                "No commands to memorize"
              ]}
              typingSpeed={75}
              pauseDuration={2000}
              deletingSpeed={50}
              showCursor
              cursorCharacter="_"
            />
          </div>

          <p className="text-base sm:text-xl text-white/70 mb-8 sm:mb-12">
            drift web — coming soon
          </p>

          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center items-center w-full sm:w-auto">
            <Link
              href="/app"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 bg-white text-black rounded-full text-base sm:text-lg font-semibold hover:bg-white/90 transition-all shadow-2xl active:scale-[0.98]"
            >
              Try web app
            </Link>
            <Link
              href="/app"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 bg-white/10 border-2 border-white/30 text-white rounded-full text-base sm:text-lg font-medium hover:bg-white/20 transition-all active:scale-[0.98]"
            >
              Open web app
            </Link>
            <Link
              href="/drift"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 border-2 border-white/30 text-white rounded-full text-base sm:text-lg font-medium hover:bg-white/10 transition-all active:scale-[0.98]"
            >
              Get drift CLI
            </Link>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 border-2 border-white/30 text-white rounded-full text-base sm:text-lg font-medium hover:bg-white/10 transition-all active:scale-[0.98] inline-flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
              GitHub
            </a>
          </div>

          <p className="text-sm text-white/50 mt-8 sm:mt-12">
            Open source · MIT licensed · by Lakshit Sachdeva
          </p>

        </div>
      </main>

      <footer className="border-t border-white/10 py-6 sm:py-8 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 text-center">
          <p className="text-sm text-white/50">
            Local-first ML engineer. Same engine as the CLI. &copy; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
