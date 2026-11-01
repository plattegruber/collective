<script lang="ts">
	import { resolve } from '$app/paths';
	import Logo from '#lib/components/Logo.svelte';
	import type { LayoutProps } from './$types';

	let { data, children }: LayoutProps = $props();
</script>

{#if !data.gallery.isPublic}
	<p class="bg-accent-soft px-4 py-2 text-center text-xs text-accent-deep">
		Only you can see this gallery. Turn on sharing in Settings to publish it.
	</p>
{/if}
{@render children()}
<footer class="px-6 py-10 text-center text-xs text-ink-mute">
	<a href={resolve('/')} class="inline-flex items-center gap-2 rounded-lg focus-visible:focus-ring"
		><Logo class="opacity-70" /></a
	>
	<p class="mt-2">An inventory for the art you live with.</p>
</footer>
