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
  <div class="camera-overlay" role="dialog" aria-modal="true">
    <div class="layer base"></div>
    <div class="layer blob-top"></div>
    <div class="layer blob-bottom"></div>
    <div class="layer vignette-top"></div>
    <div class="layer vignette-bottom"></div>
    <div class="layer vignette-left"></div>
    <div class="layer vignette-right"></div>

    <div class="camera-content">
      <div class="card">
        <h2>Let’s turn on your camera</h2>
        <p class="lead">
          We only use it live to line the artwork up with the wall. No photos are captured or stored.
        </p>

        {#if message}
          <p class="detail">{message}</p>
        {/if}
        {#if error}
          <p class="detail error">{error}</p>
        {/if}

        {#if canRequest}
          <button
            class="action"
            type="button"
            onclick={handleRequest}
            disabled={isRequesting}
          >
            {isRequesting ? 'Opening camera…' : 'Enable Camera'}
          </button>
        {/if}

        <p class="privacy">We never save or send camera frames.</p>
      </div>
    </div>
  </div>
{/if}

<style>
  .camera-overlay {
    position: fixed;
    inset: 0;
    z-index: 70;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    color: #f8fbff;
    text-align: center;
  }

  .layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .layer.base {
    background: rgb(7 10 16 / 0.96);
  }

  .layer.blob-top {
    top: -120px;
    left: -80px;
    width: 340px;
    height: 340px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.55), rgba(99, 102, 241, 0.5));
    filter: blur(90px);
  }

  .layer.blob-bottom {
    bottom: -120px;
    right: -60px;
    width: 300px;
    height: 300px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.45), rgba(6, 182, 212, 0.45));
    filter: blur(90px);
  }

  .layer.vignette-top {
    inset-inline: 0;
    height: 160px;
    background: linear-gradient(to bottom, rgba(0, 0, 0, 0.7), transparent);
  }

  .layer.vignette-bottom {
    inset-inline: 0;
    height: 160px;
    bottom: 0;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.65), transparent);
  }

  .layer.vignette-left {
    inset-block: 0;
    width: 160px;
    background: linear-gradient(to right, rgba(0, 0, 0, 0.6), transparent);
  }

  .layer.vignette-right {
    inset-block: 0;
    width: 160px;
    right: 0;
    background: linear-gradient(to left, rgba(0, 0, 0, 0.6), transparent);
  }

  .camera-content {
    position: relative;
    padding: 24px;
    width: min(94vw, 420px);
  }

  .card {
    padding: clamp(28px, 4vw, 36px);
    border-radius: 28px;
    background: rgba(17, 22, 33, 0.82);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 28px 55px rgba(0, 0, 0, 0.45);
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  h2 {
    margin: 0;
    font-size: clamp(1.8rem, 5vw, 2.4rem);
    font-weight: 600;
    letter-spacing: 0.01em;
  }

  .lead {
    margin: 0;
    font-size: clamp(0.95rem, 2.8vw, 1.05rem);
    color: rgba(240, 244, 255, 0.82);
    line-height: 1.5;
  }

  .detail {
    margin: 0;
    font-size: 0.9rem;
    color: rgba(226, 232, 255, 0.75);
  }

  .detail.error {
    color: rgba(255, 112, 148, 0.9);
  }

  .action {
    margin: 10px auto 0;
    padding: 12px 26px;
    border-radius: 999px;
    border: none;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(213, 255, 248, 0.9));
    color: rgba(12, 16, 24, 0.94);
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
  }

  .action:disabled {
    opacity: 0.6;
    cursor: progress;
    box-shadow: none;
  }

  .action:hover:not(:disabled) {
    transform: translateY(-1.5px);
    box-shadow: 0 20px 44px rgba(0, 0, 0, 0.4);
  }

  .privacy {
    margin: 6px 0 0;
    font-size: 0.8rem;
    color: rgba(226, 232, 255, 0.6);
  }

  @media (max-width: 480px) {
    .camera-content {
      padding: 18px;
    }
    .card {
      border-radius: 22px;
    }
  }
</style>
