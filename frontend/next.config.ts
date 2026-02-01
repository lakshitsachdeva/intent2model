import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  experimental: {
    optimizePackageImports: ['ogl'],
  },
  turbopack: {},
};

export default nextConfig;
