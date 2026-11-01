<script lang="ts">
	import { resolve } from '$app/paths';
	import ReactionBar from '#lib/scanner/ReactionBar.svelte';
	import { formatDimensions } from '#lib/utils/dimensions.js';
	import { byline } from '#lib/utils/format.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const artwork = $derived(data.artwork);
	const primary = $derived(
		artwork.images.find((image) => image.id === artwork.primaryImageId) ?? artwork.images[0]
	);
	const sub = $derived(byline(artwork.artist, artwork.year, artwork.dateText));
	const dims = $derived(formatDimensions(artwork, artwork.dimensionUnit));
</script>

<svelte:head><title>{artwork.title} · {data.gallery.title}</title></svelte:head>

<main class="mx-auto w-full max-w-5xl px-4 pt-10 pb-10 sm:px-8">
	<a
		href={resolve('/g/[handle]', { handle: data.gallery.handle })}
		class="text-xs font-semibold tracking-[0.18em] text-ink-mute uppercase hover:text-ink"
		>← {data.gallery.title}</a
	>
	<div class="mt-6 grid gap-10 lg:grid-cols-[3fr_2fr]">
		<div class="flex flex-col gap-3">
			{#if primary}
				<img
					src={`/media/${primary.key}`}
					alt={artwork.title}
					width={primary.width}
					height={primary.height}
					class="w-full rounded-card border border-line bg-paper-deep object-contain shadow-card"
				/>
			{/if}
			{#if artwork.images.length > 1}
				<div class="flex gap-2 overflow-x-auto">
					{#each artwork.images as image (image.id)}
						<img
							src={`/media/${image.thumbKey}`}
							alt=""
							class="size-16 shrink-0 rounded-lg border border-line object-cover"
							loading="lazy"
						/>
					{/each}
				</div>
			{/if}
		</div>
		<div>
			<h1 class="font-display text-3xl font-medium tracking-tight text-balance sm:text-4xl">
				{artwork.title}
			</h1>
			{#if sub}<p class="mt-1 text-lg text-ink-soft">{sub}</p>{/if}
			<p class="mt-1 text-sm text-ink-mute">{[artwork.medium, dims].filter(Boolean).join(' · ')}</p>
			{#if artwork.description}<p class="mt-6 text-base leading-7 whitespace-pre-line">
					{artwork.description}
				</p>{/if}
			{#if artwork.provenance}
				<h2 class="mt-8 text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">
					Provenance
				</h2>
				<p class="mt-2 text-sm leading-6 whitespace-pre-line text-ink-soft">{artwork.provenance}</p>
			{/if}
			<div class="mt-8 rounded-3xl bg-ink p-4 text-white">
				<p class="mb-2 text-xs font-medium tracking-wide text-white/60 uppercase">
					How does it land?
				</p>
				<ReactionBar artworkId={artwork.id} />
			</div>
		</div>
	</div>
</main>
