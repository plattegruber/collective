<script lang="ts">
	import type { Snippet } from 'svelte';

	type Props = {
		id?: string;
		eyebrow: string;
		title: string;
		description: string;
		reverse?: boolean;
		children: Snippet;
		aside: Snippet;
	};

	let { id, eyebrow, title, description, reverse = false, children, aside }: Props = $props();
</script>

<section {id} class="scroll-mt-24 py-16 sm:py-24" aria-labelledby={id ? `${id}-title` : undefined}>
	<div
		class={[
			'mx-auto grid max-w-6xl items-center gap-10 px-5 sm:px-8 lg:grid-cols-2 lg:gap-16',
			reverse ? 'lg:[&>*:first-child]:order-2' : ''
		].join(' ')}
	>
		<div class="max-w-lg">
			<p class="text-xs font-semibold tracking-[0.18em] text-accent-deep uppercase">{eyebrow}</p>
			<h2
				id={id ? `${id}-title` : undefined}
				class="mt-3 font-display text-3xl font-medium tracking-tight text-balance sm:text-4xl"
			>
				{title}
			</h2>
			<p class="mt-4 text-base leading-7 text-ink-soft">{description}</p>
			<div class="mt-6">{@render children()}</div>
		</div>
		<div>{@render aside()}</div>
	</div>
</section>
