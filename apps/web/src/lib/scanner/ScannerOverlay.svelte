<script lang="ts" module>
	export interface ScannerArtwork {
		title: string;
		byline: string;
		medium: string;
		description: string;
	}
</script>

<script lang="ts">
	import ReactionBar from './ReactionBar.svelte';

	let {
		artwork,
		artworkId,
		reactions
	}: { artwork: ScannerArtwork; artworkId: string; reactions: boolean } = $props();
</script>

<div
	class="absolute inset-x-3 top-16 bottom-[max(0.75rem,env(safe-area-inset-bottom))] flex flex-col overflow-hidden rounded-3xl border border-white/15 bg-neutral-950/55 p-5 shadow-lift backdrop-blur-xl backdrop-saturate-150 sm:inset-x-auto sm:right-6 sm:bottom-6 sm:w-96"
>
	<div class="pointer-events-none min-h-0 flex-1 overflow-y-auto">
		<h2 class="font-display text-3xl font-medium text-balance">{artwork.title}</h2>
		{#if artwork.byline}<p class="mt-1 text-white/80">{artwork.byline}</p>{/if}
		{#if artwork.medium}<p class="text-sm text-white/60">{artwork.medium}</p>{/if}
		{#if artwork.description}<p class="mt-4 text-sm leading-6 text-white/75">
				{artwork.description}
			</p>{/if}
	</div>
	{#if reactions}
		<div class="mt-4 shrink-0">
			{#key artworkId}
				<ReactionBar {artworkId} />
			{/key}
		</div>
	{/if}
</div>
