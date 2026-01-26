import Intent2ModelWizard from "@/components/intent-2-model-wizard";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex flex-col">
      <nav className="border-b bg-white dark:bg-black/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold">i2</span>
            </div>
            <span className="font-bold text-xl tracking-tight">Intent2Model</span>
          </div>
          <div className="flex items-center space-x-4">
            <button className="text-sm font-medium hover:text-primary transition-colors">Documentation</button>
            <button className="text-sm font-medium hover:text-primary transition-colors">Community</button>
            <div className="h-6 w-px bg-border mx-2" />
            <button className="px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-full text-sm font-medium">
              Connect API
            </button>
          </div>
        </div>
      </nav>

      <main className="flex-1 py-12 px-4 sm:px-6 lg:px-8">
        <Intent2ModelWizard />
      </main>

      <footer className="border-t py-8 bg-white dark:bg-black">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Built with ❤️ for autonomous machine learning. &copy; 2026 Intent2Model Inc.
          </p>
        </div>
      </footer>
    </div>
  );
}
