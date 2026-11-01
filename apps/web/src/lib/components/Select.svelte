<script lang="ts">
	import type { HTMLSelectAttributes } from 'svelte/elements';
	import type { Snippet } from 'svelte';

	type Props = { invalid?: boolean; class?: string; children: Snippet } & Omit<
		HTMLSelectAttributes,
		'class'
	>;

	let {
		invalid = false,
		class: className = '',
		value = $bindable(),
		children,
		...rest
	}: Props = $props();
</script>

<select
	bind:value
	aria-invalid={invalid || undefined}
	class={[
		'h-10 w-full rounded-xl border bg-card px-3 text-sm text-ink shadow-none transition',
		'focus:border-ink/40 focus:ring-2 focus:ring-accent/30 focus:outline-none',
		invalid ? 'border-danger/70' : 'border-line',
		className
	].join(' ')}
	{...rest}
>
	{@render children()}
</select>
