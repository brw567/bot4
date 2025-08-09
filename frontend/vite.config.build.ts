import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Production build config that skips TypeScript checking
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        // Ignore certain warnings
        if (warning.code === 'UNUSED_EXTERNAL_IMPORT') return
        if (warning.code === 'CIRCULAR_DEPENDENCY') return
        warn(warning)
      }
    },
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  }
})