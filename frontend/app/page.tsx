'use client';

import Link from "next/link";
import dynamic from "next/dynamic";
import TextType from "@/components/TextType";

const Prism = dynamic(() => import("@/components/Prism"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 bg-black" />,
});

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
            <Link href="/drift" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              CLI
            </Link>
            <Link href="/drift#setup" className="text-sm font-medium text-white/80 hover:text-white transition-colors">
              Docs
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
              href="/drift#run-locally"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 bg-white text-black rounded-full text-base sm:text-lg font-semibold hover:bg-white/90 transition-all shadow-2xl active:scale-[0.98]"
            >
              Try locally
            </Link>
            <Link
              href="/drift"
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 border-2 border-white/30 text-white rounded-full text-base sm:text-lg font-medium hover:bg-white/10 transition-all active:scale-[0.98]"
            >
              Get drift CLI
            </Link>
          </div>

          <p className="text-sm text-white/50 mt-8 sm:mt-12">
            by Lakshit Sachdeva
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
