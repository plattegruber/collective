<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import logoUrl from '../assets/logo.png';

  const {
    title = 'Gallery Guide',
    status = '',
    ready = false,
    minDurationMs = 3000,
    fadeMs = 350,
  } = $props();

  const dispatch = createEventDispatcher();

  const LOGO_APPEAR_DELAY = 120;

  let mountedAt = 0;
  let hiding = $state(false);
  let entering = $state(true);

  onMount(() => {
    mountedAt = Date.now();
    requestAnimationFrame(() => {
      setTimeout(() => {
        entering = false;
      }, LOGO_APPEAR_DELAY);
    });
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

<div
  class="splash"
  aria-hidden="true"
  class:hiding={hiding}
  style={`--fade-duration:${fadeMs}ms`}
>
  <div class="splash-inner" class:hiding={hiding}>
    <img
      class="splash-logo"
      src={logoUrl}
      alt={`${title} logo`}
      decoding="async"
      class:entering={entering}
      class:hiding={hiding}
    />

    {#if status}
      <p class="splash-status">{status}</p>
    {/if}
  </div>
</div>

<style>
  .splash {
    position: fixed;
    inset: 0;
    z-index: 60;
    display: grid;
    place-items: center;
    padding: 2rem;
    background: #97d8c4;
    opacity: 1;
    transform: translateX(0);
    transition: opacity var(--fade-duration, 350ms) ease, transform var(--fade-duration, 350ms) ease;
  }

  .splash.hiding {
    opacity: 0;
    transform: translateX(1.25rem);
  }

  .splash-inner {
    max-width: 24rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    text-align: center;
    color: #0f2a23;
    opacity: 1;
    transform: translateX(0);
    transition: opacity var(--fade-duration, 350ms) ease, transform var(--fade-duration, 350ms) ease;
  }

  .splash-inner.hiding {
    opacity: 0;
    transform: translateX(2rem);
  }

  .splash-logo {
    inline-size: clamp(170px, 42vw, 280px);
    aspect-ratio: 1 / 1;
    object-fit: contain;
    object-position: center;
    opacity: 1;
    transform: translateX(0);
    transition: opacity var(--fade-duration, 350ms) ease, transform var(--fade-duration, 350ms) ease;
  }

  .splash-logo.entering {
    opacity: 0;
    transform: translateX(-1.5rem);
  }

  .splash-logo.hiding {
    opacity: 0;
    transform: translateX(1.5rem);
  }

  .splash-status {
    margin: 0.5rem 0 0;
    font-size: 0.95rem;
    color: rgba(15, 42, 35, 0.68);
  }

  @media (max-width: 480px) {
    .splash-inner {
      gap: 0.75rem;
    }
  }
</style>
