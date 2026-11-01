<script lang="ts">
	import { ScanLine } from 'lucide-svelte';
	import { Button, EmptyState } from '#lib/components/index.js';
	import Scanner from '#lib/scanner/Scanner.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head><title>Scan · Collective</title></svelte:head>

{#if data.manifest}
	<Scanner
		manifest={data.manifest}
		artworks={data.artworks}
		backHref="/app"
		title="Your collection"
	/>
{:else}
	<EmptyState
		title="No recognizer yet"
		description={`Train one from your photos to scan your walls. ${data.photographed} pieces have photos ready.`}
	>
		{#snippet icon()}<ScanLine class="size-6" aria-hidden="true" />{/snippet}
		<Button href="/app/settings#recognizer">How to train</Button>
	</EmptyState>
{/if}
