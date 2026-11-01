<script lang="ts">
	import type { Snippet } from 'svelte';

	type Props = {
		label: string;
		name: string;
		hint?: string;
		error?: string;
		optional?: boolean;
		children: Snippet<[{ id: string; describedBy: string | undefined; invalid: boolean }]>;
	};

	let { label, name, hint, error, optional = false, children }: Props = $props();

	const id = $derived(`field-${name}`);
	const hintId = $derived(`${id}-hint`);
	const errorId = $derived(`${id}-error`);
	const describedBy = $derived(
		[hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined
	);
</script>

<div class="flex flex-col gap-1.5">
	<label for={id} class="flex items-baseline justify-between text-sm font-medium text-ink">
		<span>{label}</span>
		{#if optional}
			<span class="text-xs font-normal text-ink-mute">Optional</span>
		{/if}
	</label>
	{@render children({ id, describedBy, invalid: Boolean(error) })}
	{#if error}
		<p id={errorId} class="text-sm text-danger" role="alert">{error}</p>
	{:else if hint}
		<p id={hintId} class="text-xs text-ink-mute">{hint}</p>
	{/if}
</div>
