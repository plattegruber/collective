<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchReactionCounts, sendReaction } from '#lib/reactions/client.js';
	import {
		REACTION_EMOJIS,
		type ReactionCount,
		type ReactionEmoji
	} from '#lib/reactions/shared.js';

	let { artworkId }: { artworkId: string } = $props();

	let counts = $state<ReactionCount[]>(REACTION_EMOJIS.map((emoji) => ({ emoji, count: 0 })));
	let sending = $state<ReactionEmoji | null>(null);
	let message = $state('');
	let floating = $state<{ id: number; emoji: string }[]>([]);
	let nextId = 0;

	onMount(async () => {
		counts = (await fetchReactionCounts(artworkId)).length
			? await fetchReactionCounts(artworkId)
			: counts;
	});

	async function reactWith(emoji: ReactionEmoji) {
		if (sending) return;
		sending = emoji;
		const id = nextId++;
		floating = [...floating, { id, emoji }];
		setTimeout(() => (floating = floating.filter((item) => item.id !== id)), 1300);
		try {
			const result = await sendReaction(artworkId, emoji);
			if (result.counts) counts = result.counts;
			message = result.throttled ? 'Already counted — thanks!' : 'Sent!';
		} catch (cause) {
			message = cause instanceof Error ? cause.message : 'Could not send.';
		} finally {
			sending = null;
			setTimeout(() => (message = ''), 1800);
		}
	}
</script>

<div class="relative">
	<div
		class="pointer-events-none absolute inset-x-0 -top-8 flex justify-center gap-2"
		aria-hidden="true"
	>
		{#each floating as item (item.id)}
			<span class="animate-[float_1.2s_ease-out_forwards] text-2xl">{item.emoji}</span>
		{/each}
	</div>
	<div
		class="flex items-center justify-between gap-1"
		role="group"
		aria-label="React to this piece"
	>
		{#each counts as { emoji, count } (emoji)}
			<button
				type="button"
				class="flex flex-1 flex-col items-center gap-0.5 rounded-2xl bg-white/10 py-2 text-xl transition hover:bg-white/20 focus-visible:focus-ring active:scale-95 disabled:opacity-60"
				onclick={() => reactWith(emoji)}
				disabled={sending !== null}
				aria-label={`React with ${emoji}`}
			>
				<span>{emoji}</span>
				<span class="text-[10px] font-medium text-white/60 tabular-nums">{count}</span>
			</button>
		{/each}
	</div>
	<p
		class="mt-2 h-4 text-center text-[11px] tracking-wide text-white/60 uppercase"
		aria-live="polite"
	>
		{message}
	</p>
</div>

<style>
	@keyframes float {
		0% {
			opacity: 0;
			transform: translateY(8px) scale(0.9);
		}
		20% {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
		100% {
			opacity: 0;
			transform: translateY(-56px) scale(1.05);
		}
	}
</style>
