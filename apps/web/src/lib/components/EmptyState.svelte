<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title,
		description,
		icon,
		children
	}: { title: string; description?: string; icon?: Snippet; children?: Snippet } = $props();
</script>

<div
	class="flex flex-col items-center justify-center rounded-card border border-dashed border-line bg-card/60 px-6 py-14 text-center"
>
	{#if icon}
		<div class="mb-4 grid size-12 place-items-center rounded-2xl bg-accent-soft text-accent-deep">
			{@render icon()}
		</div>
	{/if}
	<h2 class="font-display text-2xl font-medium text-ink">{title}</h2>
	{#if description}
		<p class="mt-2 max-w-md text-sm text-ink-soft">{description}</p>
	{/if}
	{#if children}
		<div class="mt-6 flex flex-wrap justify-center gap-3">{@render children()}</div>
	{/if}
</div>
