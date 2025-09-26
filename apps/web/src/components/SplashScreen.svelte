<script>
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';

  const {
    images = [],
    title = 'Gallery Guide',
    subtitle = null,
    dismissible = true,
    autoDismissMs = null,
    overlayOpacity = 0.5,
  } = $props();

  const dispatch = createEventDispatcher();

  const fallback = [
    'https://images.unsplash.com/photo-1517694712202-14dd9538aa97?q=80&w=800&auto=format&fit=crop',
    'https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?q=80&w=800&auto=format&fit=crop',
    'https://images.unsplash.com/photo-1515387784663-e2b89127fba1?q=80&w=800&auto=format&fit=crop',
    'https://images.unsplash.com/photo-1488521787991-ed7bbaae773c?q=80&w=800&auto=format&fit=crop',
    'https://images.unsplash.com/photo-1516802273409-68526ee1bdd6?q=80&w=800&auto=format&fit=crop',
    'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?q=80&w=800&auto=format&fit=crop',
  ];

  let timer = null;

  onMount(() => {
    if (autoDismissMs && autoDismissMs > 0) {
      timer = setTimeout(() => dispatch('done'), autoDismissMs);
    }
  });

  onDestroy(() => {
    if (timer) clearTimeout(timer);
  });

  const TILE_COUNT = 56;
  const pool = $derived(images?.length ? images : fallback);
  const tiles = $derived(Array.from({ length: TILE_COUNT }, (_, index) => pool[index % pool.length]));

  function proceed() {
    dispatch('done');
  }
</script>

<div class="fixed inset-0 z-[60] bg-black">
  <div
    class="absolute inset-0 grid gap-0.5 grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12"
  >
    {#each tiles as src, index}
      <div class="relative aspect-square overflow-hidden">
        <img
          src={src}
          alt=""
          class="h-full w-full object-cover will-change-transform transition-transform duration-[8000ms] ease-in-out [animation:kenburns_10s_ease-in-out_infinite_alternate]"
          style={`animation-delay:${(index % 8) * 150}ms`}
          loading="lazy"
          decoding="async"
        />
        <div class="absolute inset-0 bg-black/20"></div>
      </div>
    {/each}
  </div>

  <div class="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-black/70 to-transparent"></div>
  <div class="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black/70 to-transparent"></div>
  <div class="pointer-events-none absolute inset-y-0 left-0 w-40 bg-gradient-to-r from-black/70 to-transparent"></div>
  <div class="pointer-events-none absolute inset-y-0 right-0 w-40 bg-gradient-to-l from-black/70 to-transparent"></div>

  <div class="absolute inset-0 flex items-center justify-center p-6">
    <div
      class="relative flex max-w-[92vw] flex-col items-center gap-3 rounded-3xl px-8 py-6 text-center backdrop-blur-md shadow-2xl"
      style={`background-color: rgba(0,0,0,${overlayOpacity})`}
    >
      <h1 class="text-4xl font-semibold tracking-wide text-white sm:text-5xl md:text-6xl">
        {title}
      </h1>
      {#if subtitle}
        <p class="text-sm text-white/80 sm:text-base">{subtitle}</p>
      {/if}

      {#if dismissible}
        <button
          class="mt-4 inline-flex items-center gap-2 rounded-full bg-white/95 px-5 py-2.5 text-sm font-medium text-black shadow-lg transition hover:bg-white active:scale-[0.98]"
          type="button"
          onclick={proceed}
          aria-label="Enter Gallery Guide"
        >
          <span>Enter</span>
          <span class="inline-block h-2 w-2 animate-pulse rounded-full bg-black/70"></span>
        </button>
      {/if}

      {#if !dismissible && !autoDismissMs}
        <p class="mt-2 text-xs text-white/60">Loading…</p>
      {/if}
    </div>
  </div>
</div>

<style>
  @keyframes kenburns {
    0% {
      transform: scale(1);
    }
    100% {
      transform: scale(1.06);
    }
  }
</style>
