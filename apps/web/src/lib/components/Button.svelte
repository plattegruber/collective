<script lang="ts">
	import type { HTMLAnchorAttributes, HTMLButtonAttributes } from 'svelte/elements';
	import type { Snippet } from 'svelte';

	type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'inverse';
	type Size = 'sm' | 'md' | 'lg';

	type Props = {
		variant?: Variant;
		size?: Size;
		href?: string;
		loading?: boolean;
		children: Snippet;
		class?: string;
	} & Omit<HTMLButtonAttributes, 'class'> &
		Omit<HTMLAnchorAttributes, 'class' | 'href'>;

	let {
		variant = 'primary',
		size = 'md',
		href,
		loading = false,
		children,
		class: className = '',
		disabled,
		type = 'button',
		...rest
	}: Props = $props();

	const variants: Record<Variant, string> = {
		primary: 'bg-ink text-paper hover:bg-ink/90 shadow-card',
		secondary: 'bg-card text-ink border border-line hover:border-ink/30 hover:bg-paper-deep/60',
		ghost: 'text-ink-soft hover:bg-paper-deep hover:text-ink',
		danger: 'bg-danger text-white hover:bg-danger/90',
		inverse: 'bg-paper text-ink hover:bg-white shadow-lift'
	};
	const sizes: Record<Size, string> = {
		sm: 'h-8 px-3 text-sm gap-1.5 rounded-lg',
		md: 'h-10 px-4 text-sm gap-2 rounded-xl',
		lg: 'h-12 px-6 text-base gap-2 rounded-2xl'
	};

	const classes = $derived(
		[
			'inline-flex items-center justify-center font-medium whitespace-nowrap transition-[background-color,border-color,transform,box-shadow] duration-150 select-none',
			'focus-visible:focus-ring active:translate-y-px',
			'disabled:pointer-events-none disabled:opacity-50',
			variants[variant],
			sizes[size],
			className
		].join(' ')
	);
</script>

{#if href}
	<a {href} class={classes} aria-disabled={disabled || loading || undefined} {...rest}>
		{@render children()}
	</a>
{:else}
	<button
		{type}
		class={classes}
		disabled={disabled || loading}
		aria-busy={loading || undefined}
		{...rest}
	>
		{#if loading}
			<span
				class="size-4 animate-spin rounded-full border-2 border-current border-t-transparent"
				aria-hidden="true"
			></span>
		{/if}
		{@render children()}
	</button>
{/if}
