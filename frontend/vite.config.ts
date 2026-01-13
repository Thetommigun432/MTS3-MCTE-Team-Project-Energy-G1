import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      // Proxy local API requests to the backend server
      "/api/local": {
        target: "http://localhost:3001",
        changeOrigin: true,
      },
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    sourcemap: true,
  },
  esbuild: {
    // Strip console logs and debugger statements in production builds
    drop: mode === "production" ? ["console", "debugger"] : [],
  },
}));
