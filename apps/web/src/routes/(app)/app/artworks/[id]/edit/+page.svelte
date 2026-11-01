<script lang="ts">
	import { page } from '$app/state';
	import ArtworkForm from '#lib/components/ArtworkForm.svelte';
	import PageHeader from '#lib/components/PageHeader.svelte';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();
</script>

<svelte:head><title>Edit {data.title} · Collective</title></svelte:head>

<PageHeader eyebrow="Edit" title={data.title} />

<div class="mt-8 max-w-3xl">
	<ArtworkForm
		values={form?.values ?? data.values}
		errors={form?.errors}
		formError={form?.formError}
		submitLabel="Save changes"
		cancelHref={`/app/artworks/${page.params.id}`}
		tagSuggestions={data.tagSuggestions}
	/>
</div>
