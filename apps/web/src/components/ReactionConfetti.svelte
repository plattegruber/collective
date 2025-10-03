<script>
  const { active = false, counts = [], seed = 0 } = $props();

  const MIN_CONFETTI = 24;
  const MAX_CONFETTI = 140;
  const FALLBACK_EMOJIS = ['❤️', '👀', '🤔', '😮', '😂', '🔥'];

  let pieces = $state([]);

  function createRng(seedValue) {
    let value = seedValue || 1;
    return () => {
      value += 0x6d2b79f5;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function pickWeighted(items, random) {
    const total = items.reduce((sum, item) => sum + Math.max(0, item.count ?? 0), 0);
    if (total <= 0) {
      const pool = items.length > 0 ? items : FALLBACK_EMOJIS.map((emoji) => ({ emoji, count: 1 }));
      return pool[Math.floor(random() * pool.length)] ?? pool[0];
    }

    const threshold = random() * total;
    let running = 0;
    for (const item of items) {
      const value = Math.max(0, item.count ?? 0);
      running += value;
      if (threshold <= running) {
        return item;
      }
    }
    return items[items.length - 1];
  }

  function buildPieces(data, seedValue) {
    const list = Array.isArray(data) && data.length > 0
      ? data.filter((item) => typeof item?.emoji === 'string')
      : FALLBACK_EMOJIS.map((emoji) => ({ emoji, count: 1 }));

    const rng = createRng(Number(seedValue) || 1);
    const totalCount = list.reduce((sum, item) => sum + Math.max(0, item.count ?? 0), 0);
    const target = totalCount > 0
      ? Math.min(MAX_CONFETTI, Math.max(MIN_CONFETTI, totalCount))
      : MIN_CONFETTI;

    const items = [];
    for (let index = 0; index < target; index += 1) {
      const selection = pickWeighted(list, rng) ?? { emoji: '✨', count: 1 };
      const left = rng() * 100;
      const delay = rng() * 0.5;
      const duration = 1.8 + rng() * 1.2;
      const drift = (rng() - 0.5) * 38;
      const fallExtra = rng() * 24;
      const scale = 0.85 + rng() * 0.75;
      const spin = (rng() - 0.5) * 540;
      const startRotation = (rng() - 0.5) * 180;
      const size = 1.2 + rng() * 0.8;

      items.push({
        id: `${selection.emoji}-${index}-${Math.floor(rng() * 1_000_000)}`,
        emoji: selection.emoji,
        left,
        delay,
        duration,
        drift: `${drift.toFixed(2)}vw`,
        fall: `${fallExtra.toFixed(2)}vh`,
        scale: scale.toFixed(2),
        spin: `${spin.toFixed(1)}deg`,
        start: `${startRotation.toFixed(1)}deg`,
        size: size.toFixed(2),
      });
    }

    return items;
  }

  $effect(() => {
    if (!active) {
      pieces = [];
      return;
    }
    pieces = buildPieces(counts, seed);
  });
</script>

<div class={`reaction-confetti ${active ? 'reaction-confetti--active' : ''}`} aria-hidden="true">
  {#each pieces as piece (piece.id)}
    <span
      class="reaction-confetti__piece"
      style={`left:${piece.left}%;font-size:${piece.size}rem;animation-delay:${piece.delay}s;animation-duration:${piece.duration}s;--drift-x:${piece.drift};--fall-extra:${piece.fall};--scale:${piece.scale};--spin:${piece.spin};--start-rotation:${piece.start}`}
    >
      {piece.emoji}
    </span>
  {/each}
</div>

<style>
  .reaction-confetti {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 24;
    overflow: hidden;
    opacity: 0;
    transition: opacity 0.25s ease;
  }

  .reaction-confetti--active {
    opacity: 1;
  }

  .reaction-confetti__piece {
    position: absolute;
    top: -12%;
    transform: translate(-50%, -120%) scale(var(--scale, 1));
    animation-name: reactionConfettiFall;
    animation-timing-function: cubic-bezier(0.26, 0.01, 0.18, 1);
    animation-fill-mode: forwards;
    filter: drop-shadow(0 18px 40px rgba(0, 0, 0, 0.28));
    will-change: transform, opacity;
    opacity: 0;
  }

  @keyframes reactionConfettiFall {
    0% {
      transform: translate(-50%, -120%) scale(var(--scale, 1)) rotate(var(--start-rotation, 0deg));
      opacity: 0;
    }
    12% {
      opacity: 1;
    }
    100% {
      transform: translate(calc(-50% + var(--drift-x, 0vw)), calc(120vh + var(--fall-extra, 0vh)))
        scale(var(--scale, 1)) rotate(calc(var(--start-rotation, 0deg) + var(--spin, 0deg)));
      opacity: 0;
    }
  }
</style>
