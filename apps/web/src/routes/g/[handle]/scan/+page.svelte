<script lang="ts">
	import Scanner from '#lib/scanner/Scanner.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head><title>Scan · {data.gallery.title}</title></svelte:head>

<Scanner
	manifest={data.manifest}
	artworks={data.artworks}
	backHref={`/g/${data.gallery.handle}`}
	title={data.gallery.title}
/>
