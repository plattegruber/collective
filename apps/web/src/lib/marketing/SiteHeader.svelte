<script lang="ts">
	import { resolve } from '$app/paths';
	import Button from '#lib/components/Button.svelte';
	import Logo from '#lib/components/Logo.svelte';

	let { signedIn = false }: { signedIn?: boolean } = $props();

	const links = [
		{ href: '#inventory', label: 'Inventory' },
		{ href: '#scanner', label: 'Scanner' },
		{ href: '#how', label: 'How it works' },
		{ href: '#faq', label: 'FAQ' }
	];
</script>

<header class="sticky top-0 z-40 border-b border-line/70 bg-paper/85 backdrop-blur">
	<div class="mx-auto flex h-16 max-w-6xl items-center justify-between gap-6 px-5 sm:px-8">
		<a href={resolve('/')} class="rounded-lg focus-visible:focus-ring" aria-label="Collective home"
			><Logo /></a
		>
		<!-- Same-page anchors are not routes, so resolve() does not apply. -->
		<!-- eslint-disable svelte/no-navigation-without-resolve -->
		<nav class="hidden items-center gap-6 md:flex" aria-label="Sections">
			{#each links as link (link.href)}
				<a
					href={link.href}
					class="rounded-md text-sm font-medium text-ink-soft transition hover:text-ink focus-visible:focus-ring"
				>
					{link.label}
				</a>
			{/each}
		</nav>
		<!-- eslint-enable svelte/no-navigation-without-resolve -->
		<div class="flex items-center gap-2">
			{#if signedIn}
				<Button href="/app" size="sm">Open app</Button>
			{:else}
				<Button href="/login" variant="ghost" size="sm" class="hidden sm:inline-flex"
					>Sign in</Button
				>
				<Button href="/signup" size="sm">Create account</Button>
			{/if}
		</div>
	</div>
</header>
