<script>
  import { createEventDispatcher, onMount } from 'svelte';

  const {
    title = 'Gallery Guide',
    subtitle = 'Point • Discover • Remember',
    status = '',
    ready = false,
    minDurationMs = 3000,
    fadeMs = 350,
  } = $props();

  const dispatch = createEventDispatcher();

  let mountedAt = 0;
  let hiding = $state(false);

  onMount(() => {
    mountedAt = Date.now();
  });

  const minimumWait = (elapsed) => Math.max(0, minDurationMs - elapsed);

  async function maybeDismiss() {
    if (!ready || hiding) return;
    const elapsed = Date.now() - mountedAt;
    const wait = minimumWait(elapsed);
    if (wait > 0) {
      await new Promise((resolve) => setTimeout(resolve, wait));
    }
    hiding = true;
    await new Promise((resolve) => setTimeout(resolve, fadeMs));
    dispatch('done');
  }

  $effect(() => {
    void maybeDismiss();
  });
</script>

<div class="fixed inset-0 z-[60] select-none" aria-hidden="true">
  <div class="absolute inset-0 bg-neutral-950"></div>

  <div class="pointer-events-none absolute -top-32 -left-24 h-80 w-80 rounded-full blur-3xl opacity-30 bg-gradient-to-br from-fuchsia-500 to-indigo-500"></div>
  <div class="pointer-events-none absolute -bottom-24 -right-16 h-72 w-72 rounded-full blur-3xl opacity-25 bg-gradient-to-br from-emerald-400 to-cyan-500"></div>

  <div class="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-black/60 to-transparent"></div>
  <div class="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black/60 to-transparent"></div>
  <div class="pointer-events-none absolute inset-y-0 left-0 w-40 bg-gradient-to-r from-black/60 to-transparent"></div>
  <div class="pointer-events-none absolute inset-y-0 right-0 w-40 bg-gradient-to-l from-black/60 to-transparent"></div>

  <div class="absolute inset-0 grid place-items-center p-8">
    <div class="flex flex-col items-center text-center">
      <div
        class="translate-y-0 transition-transform duration-500 ease-out"
        class:opacity-0={hiding}
        class:scale-95={hiding}
      >
        <h1 class="text-4xl font-semibold tracking-wide text-white sm:text-5xl">
          {title}
        </h1>
        {#if subtitle}
          <p class="mt-2 text-sm text-white/80 sm:text-base">{subtitle}</p>
        {/if}

        {#if status}
          <p class="mt-4 text-xs text-white/60 sm:text-sm">{status}</p>
        {/if}

        <div class="mt-6 flex items-center justify-center gap-2">
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-white/80 [animation-delay:-200ms]"></span>
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-white/70"></span>
          <span class="h-1.5 w-1.5 animate-bounce rounded-full bg-white/60 [animation-delay:200ms]"></span>
        </div>
      </div>
    </div>
  </div>

  <div
    class="absolute inset-0 bg-black/0 transition-opacity"
    style={`transition-duration:${fadeMs}ms`}
    class:opacity-0={!hiding}
    class:opacity-100={hiding}
  ></div>
</div>
