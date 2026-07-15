import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5180,
    // In dev, the API runs separately on :8100; in the built app the
    // backend serves the SPA itself so requests are same-origin.
    proxy: {
      "/api": "http://localhost:8100",
    },
  },
});
