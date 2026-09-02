import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // In dev, forward the backend routes to the FastAPI server (uvicorn on 8000)
  // so the browser can POST to /annotate same-origin, no CORS setup needed.
  server: {
    proxy: {
      "/annotate": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
})
