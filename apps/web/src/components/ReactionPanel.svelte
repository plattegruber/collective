<script>
  import { createEventDispatcher, onDestroy } from 'svelte';
  import { REACTION_EMOJIS, sendReaction } from '../lib/reactions';

  const { pieceId = '', active = false } = $props();

  const dispatch = createEventDispatcher();
  let isOpen = $state(false);
  let isSending = $state(false);
  let errorMessage = $state('');
  let floating = $state([]);
  const timers = new Map();

  $effect(() => {
    if (!active) {
      isOpen = false;
      floating = [];
      errorMessage = '';
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
    }
  });

  function togglePanel() {
    if (!pieceId) return;
    isOpen = !isOpen;
    errorMessage = '';
  }

  function makeId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function spawnFloat(emoji) {
    const id = makeId();
    floating = [...floating, { id, emoji }];
    const timeout = setTimeout(() => {
      floating = floating.filter((item) => item.id !== id);
      timers.delete(id);
    }, 1400);
    timers.set(id, timeout);
  }

  async function handleEmojiSelect(emoji) {
    if (!pieceId || isSending) return;
    isSending = true;
    errorMessage = '';
    try {
      const result = await sendReaction(pieceId, emoji);
      spawnFloat(emoji);
      isOpen = false;
      dispatch('reacted', { emoji, throttled: Boolean(result?.throttled) });
    } catch (error) {
      console.error('Failed to send reaction', error);
      const message = error?.payload?.error ?? error?.message ?? 'Unable to send reaction.';
      errorMessage = message;
    } finally {
      isSending = false;
    }
  }

  onDestroy(() => {
    timers.forEach((timer) => clearTimeout(timer));
    timers.clear();
  });

  $effect(() => {
    if (!pieceId) {
      isOpen = false;
      floating = [];
    }
  });
</script>

<div class="relative flex justify-end">
  <div class="pointer-events-none absolute bottom-16 right-0 flex flex-col items-end gap-1">
    {#each floating as item (item.id)}
      <span class="floating-emoji text-2xl drop-shadow" aria-hidden="true">{item.emoji}</span>
    {/each}
  </div>

  <div class={`reaction-panel ${isOpen ? 'reaction-panel--open' : ''}`}>
    <div class="flex items-center justify-between">
      <p class="text-sm font-medium text-white/90">How does it land?</p>
      <button
        class="text-xs font-semibold uppercase tracking-wide text-white/60 transition hover:text-white"
        type="button"
        onclick={() => {
          isOpen = false;
          errorMessage = '';
        }}
      >
        Close
      </button>
    </div>
    <div class="mt-3 grid grid-cols-3 gap-3">
      {#each REACTION_EMOJIS as emoji}
        <button
          class="reaction-button"
          type="button"
          aria-label={`Send ${emoji}`}
          onclick={() => handleEmojiSelect(emoji)}
          disabled={isSending}
        >
          <span class="text-2xl">{emoji}</span>
        </button>
      {/each}
    </div>
    {#if errorMessage}
      <p class="mt-3 text-xs text-red-200/80">{errorMessage}</p>
    {/if}
  </div>

  <button
    class={`reaction-toggle ${isOpen ? 'reaction-toggle--active' : ''}`}
    type="button"
    aria-expanded={isOpen}
    onclick={togglePanel}
  >
    <span class="sr-only">React to this piece</span>
    <svg class="h-5 w-5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M10 4v12M4 10h12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
    </svg>
  </button>
</div>

<style>
  .reaction-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    inline-size: 44px;
    block-size: 44px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.16);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    transition: background 0.2s ease, transform 0.2s ease;
  }

  .reaction-toggle:hover,
  .reaction-toggle:focus-visible {
    background: rgba(255, 255, 255, 0.26);
    transform: translateY(-1px);
  }

  .reaction-toggle--active {
    background: rgba(255, 255, 255, 0.35);
  }

  .reaction-panel {
    position: absolute;
    bottom: 64px;
    right: 0;
    width: min(260px, 80vw);
    padding: 16px 18px 20px;
    border-radius: 20px;
    background: rgba(11, 13, 22, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.36);
    backdrop-filter: blur(20px) saturate(130%);
    -webkit-backdrop-filter: blur(20px) saturate(130%);
    transform: translateY(12px);
    opacity: 0;
    pointer-events: none;
    transition: transform 0.25s ease, opacity 0.25s ease;
  }

  .reaction-panel--open {
    transform: translateY(0);
    opacity: 1;
    pointer-events: auto;
  }

  .reaction-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 0;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.16);
    transition: transform 0.18s ease, background 0.18s ease;
    color: #fff;
  }

  .reaction-button:hover,
  .reaction-button:focus-visible {
    transform: translateY(-2px) scale(1.04);
    background: rgba(255, 255, 255, 0.2);
  }

  .reaction-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .floating-emoji {
    animation: emojiFloat 1.2s ease-out forwards;
  }

  @keyframes emojiFloat {
    0% {
      opacity: 0;
      transform: translateY(10px) scale(0.9);
    }
    20% {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
    80% {
      opacity: 1;
      transform: translateY(-48px) scale(1.08);
    }
    100% {
      opacity: 0;
      transform: translateY(-64px) scale(0.9);
    }
  }
</style>
