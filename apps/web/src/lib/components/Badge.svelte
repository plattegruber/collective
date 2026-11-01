<script lang="ts">
	import type { Snippet } from 'svelte';

	type Tone = 'neutral' | 'accent' | 'success' | 'danger';
	let { tone = 'neutral', children }: { tone?: Tone; children: Snippet } = $props();

	const tones: Record<Tone, string> = {
		neutral: 'bg-paper-deep text-ink-soft',
		accent: 'bg-accent-soft text-accent-deep',
		success: 'bg-success-soft text-success',
		danger: 'bg-danger-soft text-danger'
	};
</script>

<span
	class={[
		'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
		tones[tone]
	].join(' ')}
>
	{@render children()}
</span>
