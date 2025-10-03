<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import logoUrl from '../assets/logo.png';

  const {
    title = 'Gallery Guide',
    status = '',
    ready = false,
    minDurationMs = 3000,
    fadeMs = 350,
    awaitingPermission = false,
    permissionMessage = '',
    permissionCopy = 'We need camera access for this to work.',
    permissionDisabled = false,
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

  function requestCamera() {
    dispatch('cameraRequest');
  }

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

    {#if awaitingPermission}
      <div class="splash-permission" class:entering={entering} class:hiding={hiding}>
        <button
          type="button"
          class="splash-cta"
          onclick={requestCamera}
          disabled={permissionDisabled}
        >
          Enable Camera
        </button>
        <p class="splash-permission-text">{permissionCopy}</p>
        {#if permissionMessage}
          <p class="splash-permission-hint">{permissionMessage}</p>
        {/if}
        {#if status}
          <p class="splash-permission-hint splash-permission-hint--status">{status}</p>
        {/if}
      </div>
    {:else if status}
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

  .splash-permission {
    margin: 1.5rem 0 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.55rem;
    opacity: 1;
    transform: translateX(0);
    transition: opacity var(--fade-duration, 350ms) ease, transform var(--fade-duration, 350ms) ease;
  }

  .splash-permission.entering {
    opacity: 0;
    transform: translateX(-1rem);
  }

  .splash-permission.hiding {
    opacity: 0;
    transform: translateX(1rem);
  }

  .splash-cta {
    padding: 0.75rem 1.75rem;
    border-radius: 999px;
    border: none;
    background: #c6efe1;
    color: #0f2a23;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: transform 250ms ease, box-shadow 250ms ease, background-color 250ms ease;
    box-shadow: 0 10px 24px rgba(15, 42, 35, 0.2);
  }

  .splash-cta:hover {
    transform: translateY(-1px);
    background: #b5e8d7;
    box-shadow: 0 14px 30px rgba(15, 42, 35, 0.25);
  }

  .splash-cta:active {
    transform: translateY(1px);
  }

  .splash-cta:focus-visible {
    outline: 2px solid rgba(15, 42, 35, 0.45);
    outline-offset: 6px;
  }

  .splash-cta:disabled {
    cursor: default;
    opacity: 0.9;
    box-shadow: none;
  }

  .splash-permission-text {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }

  .splash-permission-hint {
    margin: 0;
    font-size: 0.95rem;
    color: rgba(15, 42, 35, 0.68);
  }

  .splash-permission-hint--status {
    color: rgba(15, 42, 35, 0.55);
  }

  .splash-status {
    margin: 1.25rem 0 0;
    font-size: 0.95rem;
    color: rgba(15, 42, 35, 0.68);
  }

  @media (max-width: 480px) {
    .splash-inner {
      gap: 0.75rem;
    }

    .splash-permission {
      gap: 0.5rem;
      margin-top: 1.25rem;
    }

    .splash-permission-text {
      font-size: 1rem;
    }

    .splash-permission-hint {
      font-size: 0.9rem;
    }

    .splash-status {
      margin-top: 1rem;
      font-size: 0.9rem;
    }
  }
</style>
