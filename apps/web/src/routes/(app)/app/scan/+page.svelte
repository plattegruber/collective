<script lang="ts">
	import { ScanLine } from 'lucide-svelte';
	import { Button, EmptyState } from '#lib/components/index.js';
	import Scanner from '#lib/scanner/Scanner.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head><title>Scan · Collective</title></svelte:head>

{#if data.manifest && data.index}
	<Scanner
		manifest={data.manifest}
		index={data.index}
		artworks={data.artworks}
		backHref="/app"
		title="Your collection"
		canTeach={data.canTeach}
	/>
{:else}
	<EmptyState
		title="The recognizer is not available yet"
		description="No recognizer release has been published for this installation. Once one is live, your photos are indexed automatically and the scanner works right away."
	>
		{#snippet icon()}<ScanLine class="size-6" aria-hidden="true" />{/snippet}
		<Button href="/app/settings#recognizer">Recognizer settings</Button>
	</EmptyState>
{/if}
