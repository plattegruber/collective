import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  base: '/collective/',
  plugins: [svelte()],
  publicDir: 'public',
  server: {
    host: '0.0.0.0',
    port: 3000,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    target: 'es2022',
    rollupOptions: {
      external: ['three'],
    },
  },
});
