<script lang="ts">
	let { class: className = '', wordmark = true }: { class?: string; wordmark?: boolean } = $props();
</script>

<span class={['inline-flex items-center gap-2', className].join(' ')}>
	<svg viewBox="0 0 32 32" class="size-7" aria-hidden="true">
		<rect x="2" y="2" width="28" height="28" rx="8" class="fill-ink" />
		<rect x="8" y="8" width="16" height="16" rx="3" class="fill-paper" />
		<rect x="11" y="11" width="10" height="10" rx="1.5" class="fill-accent" />
	</svg>
	{#if wordmark}
		<span class="font-display text-xl font-semibold tracking-tight">Collective</span>
	{/if}
</span>
