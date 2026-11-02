<script lang="ts">
	import { ArrowRight, Frame, ImageIcon, Plus, ScanLine, Sparkles } from 'lucide-svelte';
	import { Button, Card, EmptyState, PageHeader } from '#lib/components/index.js';
	import ArtworkCard from '#lib/components/ArtworkCard.svelte';
	import { formatMoney } from '#lib/utils/money.js';
	import { pluralize } from '#lib/utils/format.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const valueEntries = $derived(Object.entries(data.stats.valueByCurrency));
	const firstName = $derived(data.user.name.split(' ')[0] ?? data.user.name);
	const hour = new Date().getHours();
	const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
</script>

<svelte:head><title>Overview · Collective</title></svelte:head>

<PageHeader
	eyebrow="Overview"
	title={data.welcome ? `Welcome, ${firstName}.` : `${greeting}, ${firstName}.`}
>
	{#snippet actions()}
		<Button href="/app/artworks/new"><Plus class="size-4" aria-hidden="true" /> Add artwork</Button>
	{/snippet}
</PageHeader>

{#if data.stats.total === 0}
	<div class="mt-10">
		<EmptyState
			title="Your collection starts here"
			description="Add the first piece you own. Photos, provenance, value, where it hangs — everything in one place."
		>
			{#snippet icon()}<Frame class="size-6" aria-hidden="true" />{/snippet}
			<Button href="/app/artworks/new" size="lg"
				><Plus class="size-4" aria-hidden="true" /> Add your first artwork</Button
			>
		</EmptyState>
	</div>
{:else}
	<section class="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Collection summary">
		<Card>
			<p class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">Pieces</p>
			<p class="mt-2 font-display text-4xl font-medium">{data.stats.total}</p>
			<p class="mt-1 text-sm text-ink-soft">
				{pluralize(data.stats.owned, 'piece')} in your possession
			</p>
		</Card>
		<Card>
			<p class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">Artists</p>
			<p class="mt-2 font-display text-4xl font-medium">{data.stats.artists}</p>
			<p class="mt-1 text-sm text-ink-soft">represented across the collection</p>
		</Card>
		<Card>
			<p class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">Estimated value</p>
			{#if valueEntries.length === 0}
				<p class="mt-2 font-display text-4xl font-medium text-ink-mute">—</p>
				<p class="mt-1 text-sm text-ink-soft">Add values to see a total</p>
			{:else}
				{#each valueEntries as [currency, minor] (currency)}
					<p class="mt-2 font-display text-4xl font-medium">
						{formatMoney(minor, currency, { compact: true })}
					</p>
				{/each}
				<p class="mt-1 text-sm text-ink-soft">estimated, or purchase price when unset</p>
			{/if}
		</Card>
		<Card>
			<p class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">Photographed</p>
			<p class="mt-2 font-display text-4xl font-medium">
				{data.stats.withImages}<span class="text-xl text-ink-mute">/{data.stats.total}</span>
			</p>
			<p class="mt-1 text-sm text-ink-soft">pieces with at least one photo</p>
		</Card>
	</section>

	<section class="mt-10">
		<div class="flex items-end justify-between">
			<h2 class="font-display text-2xl font-medium">Recently added</h2>
			<a
				href="/app/artworks"
				class="inline-flex items-center gap-1 text-sm font-medium text-ink-soft hover:text-ink"
			>
				All artworks <ArrowRight class="size-4" aria-hidden="true" />
			</a>
		</div>
		<div class="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-3 xl:grid-cols-6">
			{#each data.recent as artwork (artwork.id)}
				<ArtworkCard {artwork} compact />
			{/each}
		</div>
	</section>

	<section class="mt-10 grid gap-4 lg:grid-cols-2">
		<Card class="flex flex-col gap-3">
			<div class="flex items-center gap-2 text-accent-deep">
				<ScanLine class="size-5" aria-hidden="true" />
				<h2 class="font-display text-xl font-medium text-ink">Point-and-know</h2>
			</div>
			{#if data.recognizer && data.stats.withImages > 0}
				<p class="text-sm text-ink-soft">
					{#if data.recognizer.pending > 0}
						Your recognizer knows {pluralize(data.recognizer.indexed, 'photo')}; {pluralize(
							data.recognizer.pending,
							'photo'
						)} will be indexed the next time you open the scanner.
					{:else}
						Your recognizer knows every photographed piece. Open the scanner on your phone and point
						it at a wall.
					{/if}
				</p>
				<div><Button href="/app/scan" variant="secondary">Open scanner</Button></div>
			{:else if data.recognizer}
				<p class="text-sm text-ink-soft">
					Add a photo to any piece and the scanner recognizes it on the wall, no training needed.
				</p>
				<div><Button href="/app/artworks" variant="secondary">Add photos</Button></div>
			{:else}
				<p class="text-sm text-ink-soft">
					The recognizer is not available on this installation yet.
				</p>
			{/if}
		</Card>
		<Card class="flex flex-col gap-3">
			<div class="flex items-center gap-2 text-accent-deep">
				<Sparkles class="size-5" aria-hidden="true" />
				<h2 class="font-display text-xl font-medium text-ink">Public gallery</h2>
			</div>
			{#if data.user.galleryPublic && data.user.handle}
				<p class="text-sm text-ink-soft">
					{pluralize(data.stats.publicCount, 'piece')} on your public page.
				</p>
				<div><Button href={`/g/${data.user.handle}`} variant="secondary">View gallery</Button></div>
			{:else}
				<p class="text-sm text-ink-soft">
					Share a curated view of your collection with a link. You choose which pieces appear.
				</p>
				<div><Button href="/app/settings#gallery" variant="secondary">Turn on sharing</Button></div>
			{/if}
		</Card>
	</section>

	{#if data.stats.withImages < data.stats.total}
		<p class="mt-8 flex items-center gap-2 text-sm text-ink-mute">
			<ImageIcon class="size-4" aria-hidden="true" />
			{pluralize(data.stats.total - data.stats.withImages, 'piece')} still need a photo.
		</p>
	{/if}
{/if}
