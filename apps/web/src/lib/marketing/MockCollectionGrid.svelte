<script lang="ts">
	import { Search } from 'lucide-svelte';
	import ArtTile from './ArtTile.svelte';

	const pieces = [
		{ title: 'Harbor Light', artist: 'Ines Marlow, 2019', variant: 0 },
		{ title: 'Study in Blue', artist: 'T. Okafor, 2021', variant: 1 },
		{ title: 'Untitled (Kitchen)', artist: 'Byron Anway, 2022', variant: 2 },
		{ title: 'Field Notes II', artist: 'Mira Sandoval, 2017', variant: 3 },
		{ title: 'Night Orchard', artist: 'K. Lindqvist, 2020', variant: 4 },
		{ title: 'Pink Interior', artist: 'Dana Reyes, 2023', variant: 5 }
	];
	const tags = ['print', 'painting', 'living room', 'studio'];
</script>

<div
	class="rounded-card border border-line bg-paper p-4 shadow-lift sm:p-5"
	role="img"
	aria-label="Mockup of the artworks grid with search and tag filters"
>
	<div class="flex items-center gap-2">
		<div
			class="flex h-9 flex-1 items-center gap-2 rounded-xl border border-line bg-card px-3 text-xs text-ink-mute"
		>
			<Search class="size-3.5" aria-hidden="true" />
			Search title, artist, medium, location
		</div>
		<div
			class="hidden h-9 items-center rounded-xl border border-line bg-card px-3 text-xs text-ink-soft sm:flex"
		>
			Newest first
		</div>
	</div>
	<div class="mt-3 flex flex-wrap gap-1.5">
		{#each tags as tag, index (tag)}
			<span
				class={[
					'rounded-full border px-2.5 py-0.5 text-[11px] font-medium',
					index === 1 ? 'border-ink bg-ink text-paper' : 'border-line bg-card text-ink-soft'
				].join(' ')}>{tag}</span
			>
		{/each}
	</div>
	<div class="mt-4 grid grid-cols-3 gap-3">
		{#each pieces as piece (piece.title)}
			<div class="flex flex-col gap-1.5">
				<ArtTile
					variant={piece.variant}
					class="aspect-[4/5] rounded-xl border border-line shadow-card"
				/>
				<p class="truncate text-xs font-medium text-ink">{piece.title}</p>
				<p class="-mt-1 truncate text-[11px] text-ink-soft">{piece.artist}</p>
			</div>
		{/each}
	</div>
</div>
