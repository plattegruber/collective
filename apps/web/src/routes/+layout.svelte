<script lang="ts">
	import './layout.css';
	import favicon from '#lib/assets/favicon.svg';
	import ToastHost from '#lib/components/ToastHost.svelte';
	import NavigationProgress from '#lib/components/NavigationProgress.svelte';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<NavigationProgress />
{@render children()}
<ToastHost />
