import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "intent2model",
  description: "just upload and chat",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
