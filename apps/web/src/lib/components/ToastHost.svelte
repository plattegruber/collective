<script lang="ts">
	import { fly } from 'svelte/transition';
	import { toasts } from '#lib/toasts.svelte.js';

	const tones = {
		default: 'bg-ink text-paper',
		success: 'bg-success text-white',
		error: 'bg-danger text-white'
	} as const;
</script>

<div
	class="pointer-events-none fixed inset-x-0 bottom-[max(1rem,env(safe-area-inset-bottom))] z-[100] flex flex-col items-center gap-2 px-4"
	aria-live="polite"
>
	{#each toasts.items as toast (toast.id)}
		<button
			type="button"
			class={[
				'pointer-events-auto max-w-sm rounded-full px-4 py-2 text-sm font-medium shadow-lift',
				tones[toast.tone]
			].join(' ')}
			transition:fly={{ y: 12, duration: 180 }}
			onclick={() => toasts.dismiss(toast.id)}
		>
			{toast.message}
		</button>
	{/each}
</div>
