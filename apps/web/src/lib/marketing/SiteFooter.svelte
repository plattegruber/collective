<script lang="ts">
	import { resolve } from '$app/paths';
	import Logo from '#lib/components/Logo.svelte';

	const year = new Date().getFullYear();
</script>

<footer class="border-t border-line bg-paper-deep/40">
	<div
		class="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-10 sm:flex-row sm:items-center sm:justify-between sm:px-8"
	>
		<div class="flex flex-col gap-2">
			<Logo />
			<p class="text-sm text-ink-soft">The inventory for the art you live with.</p>
		</div>
		<nav class="flex flex-wrap gap-x-6 gap-y-2 text-sm" aria-label="Footer">
			<a href="#inventory" class="text-ink-soft hover:text-ink">Inventory</a>
			<a href="#scanner" class="text-ink-soft hover:text-ink">Scanner</a>
			<a href="#faq" class="text-ink-soft hover:text-ink">FAQ</a>
			<a href={resolve('/(auth)/login')} class="text-ink-soft hover:text-ink">Sign in</a>
			<a href={resolve('/(auth)/signup')} class="font-medium text-ink">Create account</a>
		</nav>
	</div>
	<p class="mx-auto max-w-6xl px-5 pb-8 text-xs text-ink-mute sm:px-8">
		© {year} Collective. Early access.
	</p>
</footer>
