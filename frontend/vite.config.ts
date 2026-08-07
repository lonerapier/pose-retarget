import { resolve } from 'node:path'
import { defineConfig } from 'vite'

// Two pages, not one with a switch: the operator view (index.html) and the demo
// view (demo.html) load different code and want different framing. Vite serves
// both in dev without any config, but a production build only picks up the
// entry points named here.
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        demo: resolve(__dirname, 'demo.html'),
      },
    },
  },
})
