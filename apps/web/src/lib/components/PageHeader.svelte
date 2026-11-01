<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title,
		eyebrow,
		description,
		actions
	}: { title: string; eyebrow?: string; description?: string; actions?: Snippet } = $props();
</script>

<header class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
	<div class="min-w-0">
		{#if eyebrow}
			<p class="mb-1 text-xs font-semibold tracking-[0.18em] text-ink-mute uppercase">{eyebrow}</p>
		{/if}
		<h1 class="font-display text-3xl font-medium tracking-tight text-balance text-ink sm:text-4xl">
			{title}
		</h1>
		{#if description}
			<p class="mt-2 max-w-2xl text-sm text-ink-soft sm:text-base">{description}</p>
		{/if}
	</div>
	{#if actions}
		<div class="flex shrink-0 flex-wrap gap-2">{@render actions()}</div>
	{/if}
</header>
