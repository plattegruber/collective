import App from './App.svelte';

const target = document.getElementById('app');
const app = target ? App.mount(target) : null;

export default app;
