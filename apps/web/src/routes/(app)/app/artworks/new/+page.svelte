<script lang="ts">
	import ArtworkForm from '#lib/components/ArtworkForm.svelte';
	import PageHeader from '#lib/components/PageHeader.svelte';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();
</script>

<svelte:head><title>Add artwork · Collective</title></svelte:head>

<PageHeader
	eyebrow="New"
	title="Add an artwork"
	description="Only the title is required. You can add photos right after."
/>

<div class="mt-8 max-w-3xl">
	<ArtworkForm
		values={form?.values ?? {}}
		errors={form?.errors}
		formError={form?.formError}
		submitLabel="Save artwork"
		cancelHref="/app/artworks"
		tagSuggestions={data.tagSuggestions}
	/>
</div>
