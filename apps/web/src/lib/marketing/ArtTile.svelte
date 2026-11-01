<script lang="ts">
	/**
	 * A placeholder "artwork" rendered from gradients and simple shapes, so mockups
	 * look like real collections without shipping stock imagery.
	 */
	let {
		variant = 0,
		class: className = '',
		framed = false
	}: { variant?: number; class?: string; framed?: boolean } = $props();

	const palettes = [
		{
			bg: 'linear-gradient(160deg, oklch(84% 0.08 70), oklch(62% 0.12 45))',
			a: 'oklch(32% 0.08 40)',
			b: 'oklch(92% 0.05 85)'
		},
		{
			bg: 'linear-gradient(200deg, oklch(70% 0.09 230), oklch(40% 0.08 250))',
			a: 'oklch(88% 0.06 90)',
			b: 'oklch(28% 0.05 260)'
		},
		{
			bg: 'linear-gradient(180deg, oklch(93% 0.02 90), oklch(80% 0.04 80))',
			a: 'oklch(45% 0.14 30)',
			b: 'oklch(30% 0.02 60)'
		},
		{
			bg: 'linear-gradient(140deg, oklch(55% 0.12 150), oklch(32% 0.08 170))',
			a: 'oklch(90% 0.08 95)',
			b: 'oklch(20% 0.03 160)'
		},
		{
			bg: 'linear-gradient(170deg, oklch(26% 0.03 280), oklch(14% 0.02 280))',
			a: 'oklch(78% 0.15 60)',
			b: 'oklch(60% 0.12 20)'
		},
		{
			bg: 'linear-gradient(120deg, oklch(90% 0.05 20), oklch(72% 0.13 15))',
			a: 'oklch(35% 0.1 20)',
			b: 'oklch(96% 0.02 60)'
		}
	];
	const palette = $derived(palettes[Math.abs(variant) % palettes.length] ?? palettes[0]!);
	const shape = $derived(Math.abs(variant) % 4);
</script>

<div
	class={[
		'relative overflow-hidden',
		framed ? 'border-[6px] border-[oklch(30%_0.02_60)] shadow-lift' : '',
		className
	].join(' ')}
	style={`background:${palette.bg}`}
	aria-hidden="true"
>
	<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="absolute inset-0 size-full">
		{#if shape === 0}
			<circle cx="62" cy="40" r="22" fill={palette.a} opacity="0.9" />
			<rect x="10" y="66" width="80" height="6" fill={palette.b} opacity="0.8" />
		{:else if shape === 1}
			<path d="M0 78 Q 30 50 55 70 T 100 60 V 100 H 0 Z" fill={palette.a} opacity="0.85" />
			<circle cx="24" cy="26" r="9" fill={palette.b} opacity="0.9" />
		{:else if shape === 2}
			<rect x="14" y="14" width="40" height="56" fill={palette.a} opacity="0.85" />
			<rect x="46" y="38" width="40" height="48" fill={palette.b} opacity="0.75" />
		{:else}
			<path d="M10 90 L 40 20 L 60 58 L 75 35 L 92 90 Z" fill={palette.a} opacity="0.85" />
			<rect x="0" y="0" width="100" height="10" fill={palette.b} opacity="0.5" />
		{/if}
	</svg>
</div>
