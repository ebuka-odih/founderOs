import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./pages/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        surface: "#ffffff",
        muted: "#f5f7fb",
        accent: "#6366f1",
        accentSoft: "#e0e7ff",
        success: "#16a34a",
        warning: "#f97316",
        textPrimary: "#0f172a",
        textMuted: "#64748b",
        borderLight: "#e2e8f0"
      }
    }
  },
  plugins: []
};

export default config;
