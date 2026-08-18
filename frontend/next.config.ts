import type { NextConfig } from "next";

const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Proxy /api to FastAPI so the browser only ever talks to one origin. This
  // sidesteps CORS entirely in development and keeps the client code free of
  // absolute URLs.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
