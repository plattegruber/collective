<script>
  import { onDestroy } from 'svelte';
  import ReactionPanel from './ReactionPanel.svelte';

  const DEFAULT_ARTWORK = {
    id: '',
    title: '',
    byline: '',
    materials: '',
    description: '',
  };

  const { visible = false, artwork = DEFAULT_ARTWORK } = $props();

  let panelMessage = $state('');
  let messageTimer = null;

  $effect(() => {
    if (!visible) {
      panelMessage = '';
      if (messageTimer) {
        clearTimeout(messageTimer);
        messageTimer = null;
      }
    }
  });

  function setPanelMessage(value) {
    if (messageTimer) {
      clearTimeout(messageTimer);
      messageTimer = null;
    }

    panelMessage = value;
    if (value) {
      messageTimer = setTimeout(() => {
        panelMessage = '';
        messageTimer = null;
      }, 2200);
    }
  }

  onDestroy(() => {
    if (messageTimer) {
      clearTimeout(messageTimer);
    }
  });
</script>

<div class={`artwork-overlay ${visible ? 'visible' : ''}`}>
  <div class="overlay-shell">
    <div class="overlay-copy">
      <h2>{artwork.title}</h2>
      <p class="overlay-byline">{artwork.byline}</p>
      <p class="overlay-materials">{artwork.materials}</p>
      <p class="overlay-description">{artwork.description}</p>
    </div>

    <div class="overlay-actions">
      <ReactionPanel
        pieceId={artwork.id}
        active={visible && Boolean(artwork.id)}
        on:reacted={(event) => {
          if (event?.detail?.throttled) {
            setPanelMessage('Already counted—thanks!');
          } else {
            setPanelMessage('Reaction sent!');
          }
        }}
      />
      {#if panelMessage}
        <p class="reaction-toast">{panelMessage}</p>
      {/if}
    </div>
  </div>
</div>

<style>
  .artwork-overlay {
    position: fixed;
    top: 20px;
    bottom: 20px;
    left: 20px;
    right: 20px;
    color: #f5f7ff;
    padding: clamp(20px, 4vw, 28px);
    border-radius: 18px;
    z-index: 10;
    pointer-events: auto;
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.35s ease, transform 0.35s ease;
    background: rgba(15, 17, 26, 0.48);
    box-shadow: 0 28px 60px rgba(0, 0, 0, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(16px) saturate(125%);
    -webkit-backdrop-filter: blur(16px) saturate(125%);
    overflow: hidden;
  }

  @supports not (backdrop-filter: blur(1px)) {
    .artwork-overlay {
      background: rgba(12, 14, 20, 0.72);
    }
  }

  .artwork-overlay.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .artwork-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 40%);
    opacity: 0.9;
    pointer-events: none;
  }

  .overlay-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .overlay-copy {
    pointer-events: none;
    position: relative;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .artwork-overlay h2 {
    position: relative;
    font-size: clamp(1.5rem, 4vw, 2.2rem);
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.02em;
  }

  .overlay-byline,
  .overlay-materials,
  .overlay-description {
    position: relative;
    margin: 0;
    line-height: 1.65;
    color: rgba(240, 244, 255, 0.85);
  }

  .overlay-description {
    margin-top: 12px;
    font-size: 0.95rem;
    color: rgba(224, 230, 255, 0.75);
  }

  .overlay-actions {
    margin-top: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
  }

  .reaction-toast {
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
  }
</style>
