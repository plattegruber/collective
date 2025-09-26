<script>
  import { createEventDispatcher } from 'svelte';

  const {
    visible = false,
    message = '',
    error = '',
    canRequest = false,
    isRequesting = false,
  } = $props();

  const dispatch = createEventDispatcher();

  function handleRequest() {
    if (!isRequesting) dispatch('request');
  }
</script>

{#if visible}
  <div
    class="fixed inset-0 z-[70] flex items-center justify-center overflow-hidden text-center text-slate-50"
    role="dialog"
    aria-modal="true"
  >
    <div class="pointer-events-none absolute inset-0 bg-neutral-950/95"></div>
    <div class="pointer-events-none absolute -top-32 -left-20 h-[340px] w-[340px] rounded-full bg-gradient-to-br from-fuchsia-500/55 to-indigo-500/50 blur-[90px]"></div>
    <div class="pointer-events-none absolute -bottom-28 -right-16 h-[300px] w-[300px] rounded-full bg-gradient-to-br from-emerald-400/45 to-cyan-500/45 blur-[90px]"></div>
    <div class="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-black/70 to-transparent"></div>
    <div class="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black/65 to-transparent"></div>
    <div class="pointer-events-none absolute inset-y-0 left-0 w-40 bg-gradient-to-r from-black/60 to-transparent"></div>
    <div class="pointer-events-none absolute inset-y-0 right-0 w-40 bg-gradient-to-l from-black/60 to-transparent"></div>

    <div class="relative w-full max-w-[420px] px-6 sm:px-8">
      <div class="flex flex-col gap-4 rounded-3xl border border-white/15 bg-slate-900/70 p-8 shadow-[0_28px_55px_rgba(0,0,0,0.45)] backdrop-blur-2xl backdrop-saturate-[1.4] sm:p-10">
        <h2 class="text-3xl font-semibold tracking-wide sm:text-4xl">Let’s turn on your camera</h2>

        {#if message}
          <p class="text-base leading-6 text-white/80 sm:text-lg">{message}</p>
        {/if}
        {#if error}
          <p class="text-sm font-medium text-rose-300/90 sm:text-base">{error}</p>
        {/if}

        {#if canRequest}
          <button
            class="mx-auto mt-2 inline-flex items-center justify-center rounded-full bg-white/95 px-6 py-3 text-sm font-semibold tracking-wide text-slate-900 shadow-[0_18px_40px_rgba(0,0,0,0.35)] transition hover:-translate-y-0.5 hover:shadow-[0_22px_44px_rgba(0,0,0,0.4)] disabled:translate-y-0 disabled:opacity-60 disabled:shadow-none disabled:cursor-progress"
            type="button"
            onclick={handleRequest}
            disabled={isRequesting}
          >
            {isRequesting ? 'Opening camera…' : 'Enable Camera'}
          </button>
        {/if}

        <p class="text-xs text-white/55">All camera frames stay on this device. Nothing is saved.</p>
      </div>
    </div>
  </div>
{/if}
