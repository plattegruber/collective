<script>
  import { createEventDispatcher } from 'svelte';

  export let visible = false;
  export let message = '';
  export let error = '';
  export let canRequest = false;
  export let isRequesting = false;

  const dispatch = createEventDispatcher();

  function handleRequest() {
    if (!isRequesting) dispatch('request');
  }
</script>

{#if visible}
  <div
    class="pointer-events-none fixed left-1/2 top-[clamp(24px,10vh,96px)] z-60 w-[min(92vw,420px)] -translate-x-1/2"
    role="status"
    aria-live="polite"
  >
    <div class="relative pointer-events-auto overflow-hidden rounded-2xl border border-white/20 bg-slate-900/60 px-6 py-7 text-center text-slate-100 shadow-2xl backdrop-blur-xl">
      <div class="absolute inset-0 -z-10 bg-gradient-to-tr from-white/20 to-transparent"></div>
      <h2 class="text-lg font-semibold tracking-wide md:text-xl">Enable your camera</h2>
      {#if message}
        <p class="mt-2 text-sm leading-relaxed text-slate-200/80">{message}</p>
      {/if}
      {#if error}
        <p class="mt-2 text-sm font-medium text-rose-300/90">{error}</p>
      {/if}
      {#if canRequest}
        <button
          class="mt-4 inline-flex items-center justify-center rounded-full bg-white px-6 py-2 text-sm font-semibold text-slate-900 shadow-lg transition hover:-translate-y-0.5 hover:shadow-xl disabled:cursor-progress disabled:opacity-60 disabled:shadow-none"
          type="button"
          on:click={handleRequest}
          disabled={isRequesting}
        >
          {isRequesting ? 'Requesting camera…' : 'Allow camera access'}
        </button>
      {/if}
    </div>
  </div>
{/if}
