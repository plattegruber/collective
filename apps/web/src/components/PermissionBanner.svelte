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
  <div class="permission-banner" role="status" aria-live="polite">
    <div class="permission-inner">
      <h2>Enable your camera</h2>
      {#if message}
        <p>{message}</p>
      {/if}
      {#if error}
        <p class="permission-error">{error}</p>
      {/if}
      {#if canRequest}
        <button
          class="permission-button"
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

<style>
  .permission-banner {
    position: fixed;
    top: clamp(24px, 10vh, 96px);
    left: 50%;
    transform: translateX(-50%);
    width: min(92vw, 420px);
    z-index: 60;
    pointer-events: none;
  }

  .permission-inner {
    pointer-events: auto;
    backdrop-filter: blur(16px) saturate(130%);
    -webkit-backdrop-filter: blur(16px) saturate(130%);
    background: rgba(14, 18, 28, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 18px;
    padding: clamp(18px, 4vw, 28px);
    box-shadow: 0 24px 50px rgba(0, 0, 0, 0.35);
    color: rgba(240, 244, 255, 0.92);
    text-align: center;
  }

  @supports not (backdrop-filter: blur(1px)) {
    .permission-inner {
      background: rgba(11, 14, 22, 0.82);
    }
  }

  h2 {
    margin: 0 0 10px;
    font-size: clamp(1.4rem, 3.8vw, 1.8rem);
    letter-spacing: 0.01em;
  }

  p {
    margin: 6px 0;
    line-height: 1.5;
  }

  .permission-error {
    color: rgba(255, 99, 132, 0.85);
    font-size: 0.95rem;
  }

  .permission-button {
    margin-top: 16px;
    padding: 12px 20px;
    border-radius: 999px;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    color: rgba(13, 17, 26, 0.95);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(205, 218, 255, 0.85) 100%);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .permission-button:disabled {
    opacity: 0.6;
    cursor: progress;
    box-shadow: none;
  }

  .permission-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
  }
</style>
