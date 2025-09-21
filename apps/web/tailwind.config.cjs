module.exports = {
  content: ['./index.html', './src/**/*.{svelte,js,ts}'],
  theme: {
    extend: {
      dropShadow: {
        overlay: '0 10px 40px rgba(0,0,0,0.22)',
      },
    },
  },
  plugins: [],
};
