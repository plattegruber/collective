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
  class={`absolute inset-0 px-6 transition-opacity duration-200 ${visible ? 'pointer-events-auto opacity-100 visible grid place-items-center' : 'pointer-events-none opacity-0 invisible'}`}
  aria-hidden={!visible}
  on:click={handleOverlayClick}
>
  <div class="relative flex items-center justify-center drop-shadow-[0_10px_40px_rgba(0,0,0,0.22)]">
    <svg class="w-[min(70vmin,560px)] text-white/95" viewBox="0 0 100 64" role="img" aria-label="framing guides">
      <path class="fill-none stroke-white stroke-[2.4px]" d="M8 18 L8 8 L28 8" />
      <path class="fill-none stroke-white stroke-[2.4px]" d="M92 18 L92 8 L72 8" />
      <path class="fill-none stroke-white stroke-[2.4px]" d="M8 46 L8 56 L28 56" />
      <path class="fill-none stroke-white stroke-[2.4px]" d="M92 46 L92 56 L72 56" />
    </svg>
    <div
      role="button"
      tabindex="0"
      class={`relative z-10 flex max-w-[26ch] cursor-pointer items-center justify-center px-4 text-center text-lg font-semibold text-white/85 transition-opacity duration-200 ${shimmer ? 'animate-pulse' : ''}`}
      aria-live="polite"
      style={`opacity: ${phraseOpacity};`}
      on:click|stopPropagation={requestNewPhrase}
      on:keydown|stopPropagation={handlePhraseKey}
    >
      {phrase}
    </div>
  </div>
</div>
