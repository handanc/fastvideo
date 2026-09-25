import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发模式: npm run dev → http://127.0.0.1:5173 (/api 代理到 FastAPI :8000)
// 生产模式: npm run build → dist/ 由 FastAPI 托管 (fastvideo3 serve)
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
