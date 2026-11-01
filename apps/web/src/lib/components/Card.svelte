<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	type Props = { padded?: boolean; class?: string; children: Snippet } & Omit<
		HTMLAttributes<HTMLDivElement>,
		'class'
	>;

	let { padded = true, class: className = '', children, ...rest }: Props = $props();
</script>

<div
	class={[
		'rounded-card border border-line bg-card shadow-card',
		padded ? 'p-5 sm:p-6' : '',
		className
	].join(' ')}
	{...rest}
>
	{@render children()}
</div>
