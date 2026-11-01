<script lang="ts">
	import { ImageOff } from 'lucide-svelte';
	import Badge from './Badge.svelte';
	import { byline } from '#lib/utils/format.js';
	import { STATUS_LABELS } from '#lib/schemas/artwork.js';
	import type { ArtworkListItem } from '#lib/server/artworks.js';

	let {
		artwork,
		href = `/app/artworks/${artwork.id}`,
		compact = false
	}: { artwork: ArtworkListItem; href?: string; compact?: boolean } = $props();

	const thumb = $derived(artwork.primaryImage ? `/media/${artwork.primaryImage.thumbKey}` : null);
	const sub = $derived(byline(artwork.artist, artwork.year, artwork.dateText));
</script>

<a
	{href}
	class="group flex flex-col gap-2.5 rounded-card focus-visible:focus-ring"
	aria-label={`${artwork.title}${sub ? `, ${sub}` : ''}`}
>
	<div
		class="relative aspect-[4/5] overflow-hidden rounded-card border border-line bg-paper-deep shadow-card transition group-hover:shadow-lift"
	>
		{#if thumb}
			<img
				src={thumb}
				alt=""
				loading="lazy"
				decoding="async"
				class="size-full object-cover transition duration-500 group-hover:scale-[1.03]"
			/>
		{:else}
			<div class="grid size-full place-items-center text-ink-mute">
				<ImageOff class="size-6" aria-hidden="true" />
			</div>
		{/if}
		{#if artwork.status !== 'owned'}
			<div class="absolute top-2 left-2">
				<Badge tone={artwork.status === 'wishlist' ? 'accent' : 'neutral'}
					>{STATUS_LABELS[artwork.status]}</Badge
				>
			</div>
		{/if}
	</div>
	<div class="min-w-0 px-0.5">
		<p class={['truncate font-medium text-ink', compact ? 'text-sm' : 'text-base'].join(' ')}>
			{artwork.title}
		</p>
		{#if sub}
			<p class="truncate text-xs text-ink-soft">{sub}</p>
		{/if}
	</div>
</a>
