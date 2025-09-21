<script>
  import { createEventDispatcher } from 'svelte';

  export let visible = true;
  export let phrase = '';
  export let shimmer = false;
  export let phraseOpacity = 1;

  const dispatch = createEventDispatcher();

  function handleOverlayClick() {
    dispatch('hide');
  }

  function requestNewPhrase() {
    dispatch('next');
  }

  function handlePhraseKey(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      requestNewPhrase();
    }
  }
</script>

<div
  class={`overlay ${visible ? 'is-visible' : 'is-hidden'}`}
  aria-hidden={!visible}
  on:click={handleOverlayClick}
>
  <div class="hud">
    <svg class="brackets pulse" viewBox="0 0 100 64" role="img" aria-label="framing guides">
      <path d="M8 18 L8 8 L28 8" />
      <path d="M92 18 L92 8 L72 8" />
      <path d="M8 46 L8 56 L28 56" />
      <path d="M92 46 L92 56 L72 56" />
    </svg>
    <button
      type="button"
      class={`phrase ${shimmer ? 'shimmer' : ''}`}
      aria-live="polite"
      style={`opacity: ${phraseOpacity};`}
      on:click={requestNewPhrase}
      on:keydown={handlePhraseKey}
    >
      {phrase}
    </button>
  </div>
</div>

<style>
  .overlay {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: clamp(12px, 4vmin, 28px);
    pointer-events: none;
  }

  .overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(60% 60% at 50% 45%, rgba(0, 0, 0, 0) 0%, rgba(0, 0, 0, 0.35) 100%);
    pointer-events: none;
  }

  .hud {
    position: relative;
    display: grid;
    place-items: center;
    filter: drop-shadow(var(--ui-shadow));
  }

  .brackets {
    inline-size: min(70vmin, 560px);
    block-size: auto;
    opacity: 0.95;
  }

  .brackets path {
    stroke: var(--ui-fg);
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: var(--bracket-thickness);
    vector-effect: non-scaling-stroke;
  }

  @media (prefers-reduced-motion: no-preference) {
    .pulse {
      animation: pulse 2.4s ease-in-out infinite;
    }

    @keyframes pulse {
      0%,
      100% {
        opacity: 0.85;
      }

      50% {
        opacity: 1;
      }
    }
  }

  .phrase {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    text-align: center;
    letter-spacing: 0.01em;
    font-weight: 600;
    font-size: clamp(18px, 3.6vmin, 28px);
    color: var(--ui-fg-dim);
    line-height: 1.25;
    max-inline-size: min(80vw, 26ch);
    padding: 0 clamp(8px, 2vw, 16px);
    transition: opacity 0.2s ease;
  }

  .phrase.shimmer {
    background: linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.5) 0%,
      rgba(255, 255, 255, 1) 20%,
      rgba(255, 255, 255, 0.6) 40%,
      rgba(255, 255, 255, 0.85) 60%,
      rgba(255, 255, 255, 0.6) 80%,
      rgba(255, 255, 255, 0.5) 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(var(--ui-shadow));
    background-size: 200% auto;
    animation: sheen 3.5s ease-in-out infinite;
  }

  @media (prefers-reduced-motion: reduce) {
    .phrase.shimmer {
      animation: none;
      background-size: auto;
      color: var(--ui-fg-dim);
    }
  }

  @keyframes sheen {
    0% {
      background-position: 200% 0;
    }

    100% {
      background-position: -200% 0;
    }
  }

  .is-hidden {
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.28s ease, visibility 0.28s step-end;
  }

  .is-visible {
    opacity: 1;
    visibility: visible;
    transition: opacity 0.28s ease;
  }
</style>
