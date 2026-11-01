<script lang="ts">
	import type { HTMLTextareaAttributes } from 'svelte/elements';

	type Props = { invalid?: boolean; class?: string } & Omit<HTMLTextareaAttributes, 'class'>;

	let { invalid = false, class: className = '', value = $bindable(), ...rest }: Props = $props();
</script>

<textarea
	bind:value
	aria-invalid={invalid || undefined}
	class={[
		'min-h-24 w-full rounded-xl border bg-card px-3 py-2 text-sm leading-6 text-ink shadow-none transition placeholder:text-ink-mute/70',
		'focus:border-ink/40 focus:ring-2 focus:ring-accent/30 focus:outline-none',
		invalid ? 'border-danger/70' : 'border-line',
		className
	].join(' ')}
	{...rest}></textarea>
