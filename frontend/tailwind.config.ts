import type { Config } from "tailwindcss";
import tailwindcssAnimate from "tailwindcss-animate";

export default {
  darkMode: ["class"],
  // Dark-first: no class = dark theme, 'light' class = light theme
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
        energy: {
          success: "hsl(var(--energy-success))",
          "success-foreground": "hsl(var(--energy-success-foreground))",
          warning: "hsl(var(--energy-warning))",
          "warning-foreground": "hsl(var(--energy-warning-foreground))",
          "warning-bg": "hsl(var(--energy-warning-bg))",
          error: "hsl(var(--energy-error))",
          "error-foreground": "hsl(var(--energy-error-foreground))",
          "error-bg": "hsl(var(--energy-error-bg))",
          info: "hsl(var(--energy-info))",
          "info-foreground": "hsl(var(--energy-info-foreground))",
        },
        // Okabe-Ito colorblind-safe palette
        chart: {
          1: "hsl(var(--chart-1))", // Blue
          2: "hsl(var(--chart-2))", // Sky
          3: "hsl(var(--chart-3))", // Green
          4: "hsl(var(--chart-4))", // Orange
          5: "hsl(var(--chart-5))", // Vermillion
          6: "hsl(var(--chart-6))", // Purple
          other: "hsl(var(--chart-other))", // Gray
        },
        // NILM-specific colors
        confidence: {
          high: "hsl(var(--confidence-high))",
          medium: "hsl(var(--confidence-medium))",
          low: "hsl(var(--confidence-low))",
        },
        state: {
          on: "hsl(var(--state-on))",
          off: "hsl(var(--state-off))",
        },
        teal: {
          highlight: "hsl(var(--teal-highlight))",
        },
        prediction: {
          glow: "hsl(var(--prediction-glow))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(10px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-glow": {
          "0%, 100%": {
            opacity: "1",
            boxShadow: "0 0 0 0 hsl(var(--primary) / 0.4)",
          },
          "50%": {
            opacity: "0.8",
            boxShadow: "0 0 0 8px hsl(var(--primary) / 0)",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.4s ease-out",
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
      },
    },
  },
  plugins: [tailwindcssAnimate],
} satisfies Config;
