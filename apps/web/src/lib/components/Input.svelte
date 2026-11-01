<script lang="ts">
	import type { HTMLInputAttributes } from 'svelte/elements';

	type Props = { invalid?: boolean; class?: string } & Omit<HTMLInputAttributes, 'class'>;

	let { invalid = false, class: className = '', value = $bindable(), ...rest }: Props = $props();
</script>

<input
	bind:value
	aria-invalid={invalid || undefined}
	class={[
		'h-10 w-full rounded-xl border bg-card px-3 text-sm text-ink shadow-none transition placeholder:text-ink-mute/70',
		'focus:border-ink/40 focus:ring-2 focus:ring-accent/30 focus:outline-none',
		invalid ? 'border-danger/70' : 'border-line',
		className
	].join(' ')}
	{...rest}
/>
