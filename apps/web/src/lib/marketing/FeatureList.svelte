<script lang="ts">
	import { Check } from 'lucide-svelte';

	let { items }: { items: string[] } = $props();
</script>

<ul class="flex flex-col gap-2.5">
	{#each items as item (item)}
		<li class="flex items-start gap-3 text-sm text-ink">
			<span
				class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-accent-soft text-accent-deep"
			>
				<Check class="size-3" aria-hidden="true" />
			</span>
			{item}
		</li>
	{/each}
</ul>
