<script lang="ts">
	import ArtworkForm from '#lib/components/ArtworkForm.svelte';
	import PageHeader from '#lib/components/PageHeader.svelte';
	import { onMount } from 'svelte';
	import { peekCapture } from '#lib/scanner/capture.js';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();

	// Read after hydration only: the server has no sessionStorage and must render the same markup.
	let capture = $state<string | null>(null);
	onMount(() => {
		capture = peekCapture()?.dataUrl ?? null;
	});
</script>

<svelte:head><title>Add artwork · Collective</title></svelte:head>

<PageHeader
	eyebrow="New"
	title="Add an artwork"
	description="Only the title is required. You can add photos right after."
/>

<div class="mt-8 max-w-3xl">
	{#if capture}
		<div
			class="mb-6 flex items-center gap-4 rounded-card border border-accent/30 bg-accent-soft/40 p-3"
		>
			<img
				src={capture}
				alt=""
				class="size-16 shrink-0 rounded-xl border border-line object-cover"
			/>
			<p class="text-sm text-ink">
				The photo from the scanner will be attached as soon as you save.
			</p>
		</div>
	{/if}
	<ArtworkForm
		values={form?.values ?? {}}
		errors={form?.errors}
		formError={form?.formError}
		submitLabel="Save artwork"
		cancelHref="/app/artworks"
		tagSuggestions={data.tagSuggestions}
	/>
</div>
