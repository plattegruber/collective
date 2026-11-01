<script lang="ts">
	import ArtTile from './ArtTile.svelte';

	const stats = [
		{ label: 'Pieces', value: '47', sub: '44 in your possession' },
		{ label: 'Artists', value: '29', sub: 'represented across the collection' },
		{ label: 'Estimated value', value: '$86K', sub: 'estimated, or purchase price when unset' },
		{ label: 'Photographed', value: '41', sub: 'pieces with at least one photo', suffix: '/47' }
	];
</script>

<div
	class="rounded-card border border-line bg-paper p-4 shadow-lift sm:p-5"
	role="img"
	aria-label="Mockup of the overview page with collection statistics and recently added artworks"
>
	<p class="text-[10px] font-semibold tracking-[0.18em] text-ink-mute uppercase">Overview</p>
	<p class="font-display text-xl font-medium">Good evening, Andrea.</p>
	<div class="mt-4 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
		{#each stats as stat (stat.label)}
			<div class="rounded-xl border border-line bg-card p-3">
				<p class="text-[9px] font-semibold tracking-[0.16em] text-ink-mute uppercase">
					{stat.label}
				</p>
				<p class="mt-1 font-display text-2xl font-medium">
					{stat.value}{#if stat.suffix}<span class="text-sm text-ink-mute">{stat.suffix}</span>{/if}
				</p>
				<p class="mt-0.5 text-[10px] leading-4 text-ink-soft">{stat.sub}</p>
			</div>
		{/each}
	</div>
	<div class="mt-4 flex items-end justify-between">
		<p class="font-display text-base font-medium">Recently added</p>
		<p class="text-[11px] text-ink-soft">All artworks →</p>
	</div>
	<div class="mt-2 grid grid-cols-6 gap-2">
		{#each [1, 4, 0, 5, 3, 2] as variant (variant)}
			<ArtTile {variant} class="aspect-[4/5] rounded-lg border border-line" />
		{/each}
	</div>
</div>
