<script lang="ts">
	import { Plus, Search, X } from 'lucide-svelte';
	import { Button, EmptyState, Input, PageHeader, Select } from '#lib/components/index.js';
	import ArtworkCard from '#lib/components/ArtworkCard.svelte';
	import { ARTWORK_STATUSES } from '#lib/domain/artwork.js';
	import { STATUS_LABELS } from '#lib/schemas/artwork.js';
	import { pluralize } from '#lib/utils/format.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	let searchForm: HTMLFormElement | undefined = $state();
	const filtered = $derived(
		Boolean(data.filters.query || data.filters.tag || data.filters.status !== 'all')
	);
	const sortLabels = {
		newest: 'Newest first',
		oldest: 'Oldest first',
		title: 'Title A–Z',
		artist: 'Artist A–Z',
		year: 'Year, newest',
		value: 'Value, highest'
	};
</script>

<svelte:head><title>Artworks · Collective</title></svelte:head>

<PageHeader
	eyebrow="Collection"
	title="Artworks"
	description={pluralize(data.artworks.length, 'piece')}
>
	{#snippet actions()}
		<Button href="/app/export.csv" variant="ghost" download>Export CSV</Button>
		<Button href="/app/artworks/new"><Plus class="size-4" aria-hidden="true" /> Add artwork</Button>
	{/snippet}
</PageHeader>

<form
	bind:this={searchForm}
	method="GET"
	data-sveltekit-keepfocus
	data-sveltekit-replacestate
	class="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center"
>
	<div class="relative flex-1">
		<Search
			class="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-ink-mute"
			aria-hidden="true"
		/>
		<Input
			name="q"
			type="search"
			value={data.filters.query}
			placeholder="Search title, artist, medium, location"
			class="pl-9"
			aria-label="Search artworks"
		/>
	</div>
	<Select
		name="status"
		value={data.filters.status}
		aria-label="Status"
		class="sm:w-40"
		onchange={() => searchForm?.requestSubmit()}
	>
		<option value="all">All statuses</option>
		{#each ARTWORK_STATUSES as status (status)}<option value={status}
				>{STATUS_LABELS[status]}</option
			>{/each}
	</Select>
	<Select
		name="sort"
		value={data.filters.sort}
		aria-label="Sort"
		class="sm:w-44"
		onchange={() => searchForm?.requestSubmit()}
	>
		{#each Object.entries(sortLabels) as [value, label] (value)}<option {value}>{label}</option
			>{/each}
	</Select>
	{#if data.filters.tag}<input type="hidden" name="tag" value={data.filters.tag} />{/if}
	<noscript><Button type="submit" variant="secondary">Apply</Button></noscript>
</form>

{#if data.tags.length > 0}
	<div class="mt-4 flex flex-wrap items-center gap-2" aria-label="Filter by tag">
		{#each data.tags as tag (tag.id)}
			{@const active = data.filters.tag === tag.name}
			<a
				href={active ? '/app/artworks' : `/app/artworks?tag=${encodeURIComponent(tag.name)}`}
				class={[
					'inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium transition',
					active
						? 'border-ink bg-ink text-paper'
						: 'border-line bg-card text-ink-soft hover:border-ink/40 hover:text-ink'
				].join(' ')}
			>
				{tag.name}
				<span class={active ? 'text-paper/70' : 'text-ink-mute'}>{tag.count}</span>
				{#if active}<X class="size-3" aria-hidden="true" />{/if}
			</a>
		{/each}
	</div>
{/if}

{#if data.artworks.length === 0}
	<div class="mt-8">
		{#if filtered}
			<EmptyState
				title="Nothing matches"
				description="Try a different search or clear the filters."
			>
				<Button href="/app/artworks" variant="secondary">Clear filters</Button>
			</EmptyState>
		{:else}
			<EmptyState title="No artworks yet" description="Add the first piece you own.">
				<Button href="/app/artworks/new"
					><Plus class="size-4" aria-hidden="true" /> Add artwork</Button
				>
			</EmptyState>
		{/if}
	</div>
{:else}
	<div class="mt-8 grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
		{#each data.artworks as artwork (artwork.id)}
			<ArtworkCard {artwork} />
		{/each}
	</div>
{/if}
