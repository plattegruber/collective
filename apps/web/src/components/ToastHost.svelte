<script>
  import { onDestroy } from 'svelte';
  import { toasts, removeToast } from '../lib/toasts';

  const { placement = 'bottom-center' } = $props();

  let items = $state([]);
  const timers = new Map();

  const unsubscribe = toasts.subscribe((value) => {
    items = value;
    const activeIds = new Set();
    value.forEach((toast) => {
      activeIds.add(toast.id);
      if (!timers.has(toast.id) && toast.duration > 0) {
        const timeout = setTimeout(() => {
          removeToast(toast.id);
          timers.delete(toast.id);
        }, toast.duration);
        timers.set(toast.id, timeout);
      }
    });

    timers.forEach((timer, id) => {
      if (!activeIds.has(id)) {
        clearTimeout(timer);
        timers.delete(id);
      }
    });
  });

  onDestroy(() => {
    unsubscribe();
    timers.forEach((timer) => clearTimeout(timer));
    timers.clear();
  });

  function dismiss(id) {
    removeToast(id);
  }

  function placementClass(name) {
    switch (name) {
      case 'top-center':
        return 'toast-host--top-center';
      case 'top-right':
        return 'toast-host--top-right';
      case 'bottom-right':
        return 'toast-host--bottom-right';
      case 'bottom-left':
        return 'toast-host--bottom-left';
      case 'top-left':
        return 'toast-host--top-left';
      default:
        return 'toast-host--bottom-center';
    }
  }
</script>

{#if items.length > 0}
  <div class={`toast-host ${placementClass(placement)}`} aria-live="polite" aria-atomic="false">
    {#each items as toast (toast.id)}
      <div class={`toast-item toast-item--${toast.tone}`}>
        <p class="toast-item__message">{toast.message}</p>
        <button class="toast-item__close" type="button" onclick={() => dismiss(toast.id)} aria-label="Dismiss">
          ×
        </button>
      </div>
    {/each}
  </div>
{/if}

<style>
  .toast-host {
    position: fixed;
    z-index: 50;
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
  }

  .toast-host--bottom-center {
    left: 50%;
    bottom: 24px;
    transform: translateX(-50%);
    align-items: center;
  }

  .toast-host--bottom-right {
    right: 24px;
    bottom: 24px;
    align-items: flex-end;
  }

  .toast-host--bottom-left {
    left: 24px;
    bottom: 24px;
    align-items: flex-start;
  }

  .toast-host--top-center {
    left: 50%;
    top: 24px;
    transform: translateX(-50%);
    align-items: center;
  }

  .toast-host--top-right {
    right: 24px;
    top: 24px;
    align-items: flex-end;
  }

  .toast-host--top-left {
    left: 24px;
    top: 24px;
    align-items: flex-start;
  }

  .toast-item {
    pointer-events: auto;
    min-width: 220px;
    max-width: min(90vw, 320px);
    padding: 12px 16px;
    border-radius: 14px;
    background: rgba(19, 21, 32, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.32);
    color: rgba(241, 244, 255, 0.92);
    display: flex;
    gap: 10px;
    align-items: center;
    animation: toast-slide-in 0.28s ease;
  }

  .toast-item__message {
    margin: 0;
    flex: 1 1 auto;
    font-size: 0.9rem;
    line-height: 1.25;
  }

  .toast-item__close {
    appearance: none;
    border: none;
    background: none;
    color: rgba(241, 244, 255, 0.8);
    font-size: 1.2rem;
    line-height: 1;
    padding: 0;
    cursor: pointer;
    transition: color 0.2s ease;
  }

  .toast-item__close:hover,
  .toast-item__close:focus-visible {
    color: #fff;
  }

  .toast-item--error {
    border-color: rgba(255, 120, 120, 0.45);
    background: rgba(41, 18, 24, 0.94);
    color: rgba(255, 220, 224, 0.9);
  }

  .toast-item--success {
    border-color: rgba(120, 255, 180, 0.45);
    background: rgba(18, 41, 33, 0.94);
    color: rgba(212, 255, 232, 0.92);
  }

  @keyframes toast-slide-in {
    0% {
      transform: translateY(8px);
      opacity: 0;
    }

    100% {
      transform: translateY(0);
      opacity: 1;
    }
  }
</style>
