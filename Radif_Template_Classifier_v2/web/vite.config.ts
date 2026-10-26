import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // The API port, overridable so the dev server can point at either a
      // locally run uvicorn or the container, which publishes 8080.
      "/api": {
        target: process.env.DASTGAH_API ?? "http://127.0.0.1:8080",
        changeOrigin: true,
        // A long recording takes tens of seconds; the default would cut it off.
        timeout: 15 * 60 * 1000,
      },
    },
  },
});
