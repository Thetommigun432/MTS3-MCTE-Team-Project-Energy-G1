/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    // Proxy API requests to backend for local development
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/live": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ready": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/metrics": {
        target: "http://localhost:8000",
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
    // React 19 + Vite 7 targets modern browsers. 
    // ES2022 enables top-level await and newer class features.
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // React core is small; usually better to keep together unless strict caching needs
          // "vendor-react": ["react", "react-dom", "react-router-dom"],

          // Isolate heavy third-party libs to improve cache hit rates
          "vendor-supabase": ["@supabase/supabase-js"],
          "vendor-recharts": ["recharts"],
          "vendor-ui": ["@radix-ui/react-dialog", "@radix-ui/react-popover"],
        },
      },
    },
  },

  esbuild: {
    // Only drop console in production to keep dev debugging easy
    drop: mode === "production" ? ["console", "debugger"] : [],
  },

  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: false, // Disabling CSS parsing speeds up tests significantly
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/types/**',
        'dist/**',
      ],
    },
  },
}));
