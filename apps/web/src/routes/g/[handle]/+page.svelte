<script lang="ts">
	import { ScanLine } from 'lucide-svelte';
	import { Button } from '#lib/components/index.js';
	import ArtworkCard from '#lib/components/ArtworkCard.svelte';
	import { pluralize } from '#lib/utils/format.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head>
	<title>{data.gallery.title}</title>
	<meta
		name="description"
		content={data.gallery.intro ??
			`${pluralize(data.artworks.length, 'piece')} collected by ${data.gallery.ownerName}.`}
	/>
</svelte:head>

<main class="mx-auto w-full max-w-6xl px-4 pt-14 pb-10 sm:px-8">
	<header class="max-w-2xl">
		<p class="text-xs font-semibold tracking-[0.2em] text-ink-mute uppercase">
			{data.gallery.ownerName}
		</p>
		<h1 class="mt-2 font-display text-4xl font-medium tracking-tight text-balance sm:text-5xl">
			{data.gallery.title}
		</h1>
		{#if data.gallery.intro}<p class="mt-4 text-base leading-7 text-ink-soft">
				{data.gallery.intro}
			</p>{/if}
		{#if data.canScan}
			<div class="mt-6">
				<Button href={`/g/${data.gallery.handle}/scan`} size="lg"
					><ScanLine class="size-4" aria-hidden="true" /> Scan a piece</Button
				>
				<p class="mt-2 text-xs text-ink-mute">Point your phone at any piece on the wall.</p>
			</div>
		{/if}
	</header>

	{#if data.artworks.length === 0}
		<p class="mt-16 text-center text-sm text-ink-mute">Nothing on view yet.</p>
	{:else}
		<div class="mt-12 grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4">
			{#each data.artworks as artwork (artwork.id)}
				<ArtworkCard {artwork} href={`/g/${data.gallery.handle}/${artwork.id}`} />
			{/each}
		</div>
	{/if}
</main>
