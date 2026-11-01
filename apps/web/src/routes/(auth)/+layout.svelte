<script lang="ts">
	import Logo from '#lib/components/Logo.svelte';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
</script>

<div class="flex min-h-full flex-col bg-paper">
	<header class="px-6 py-5">
		<a href="/" class="inline-flex rounded-lg focus-visible:focus-ring"><Logo /></a>
	</header>
	<main class="flex flex-1 items-start justify-center px-4 pt-6 pb-16 sm:pt-12">
		<div class="w-full max-w-md">{@render children()}</div>
	</main>
</div>
