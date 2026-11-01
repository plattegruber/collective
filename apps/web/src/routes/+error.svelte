<script lang="ts">
	import { page } from '$app/state';
	import Button from '#lib/components/Button.svelte';
	import Logo from '#lib/components/Logo.svelte';

	const status = $derived(page.status);
	const title = $derived(
		status === 404
			? 'Nothing hangs here'
			: status === 403
				? 'That wall is private'
				: 'Something went wrong'
	);
	const hint = $derived(
		status === 404
			? 'The page you asked for does not exist or has been taken down.'
			: status === 403
				? 'You need to sign in as the owner to see this.'
				: (page.error?.message ?? 'An unexpected error happened. It has been noted.')
	);
</script>

<svelte:head><title>{status} · Collective</title></svelte:head>

<main class="grid min-h-full place-items-center bg-paper px-6 py-16">
	<div class="w-full max-w-md text-center">
		<a href="/" class="inline-flex rounded-lg focus-visible:focus-ring"><Logo /></a>
		<p class="mt-10 font-display text-7xl font-medium text-ink-mute/60">{status}</p>
		<h1 class="mt-3 font-display text-3xl font-medium tracking-tight text-balance">{title}</h1>
		<p class="mt-3 text-sm text-ink-soft">{hint}</p>
		<div class="mt-8 flex justify-center gap-3">
			<Button href="/" variant="secondary">Home</Button>
			<Button href="/app">Open the app</Button>
		</div>
	</div>
</main>
