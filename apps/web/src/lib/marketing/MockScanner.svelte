<script lang="ts">
	import ArtTile from './ArtTile.svelte';

	const emojis = ['❤️', '👀', '🤔', '😮', '😂', '🔥'];
</script>

<div
	class="relative mx-auto w-[260px] sm:w-[280px]"
	role="img"
	aria-label="Mockup of the phone scanner recognizing a painting and showing its story with emoji reactions"
>
	<div
		class="relative aspect-[9/19] overflow-hidden rounded-[2.6rem] border-[10px] border-[oklch(18%_0.01_280)] bg-[oklch(18%_0.01_280)] shadow-lift"
	>
		<div class="absolute top-2 left-1/2 z-20 h-5 w-24 -translate-x-1/2 rounded-full bg-black"></div>
		<div
			class="absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_0%,oklch(34%_0.02_70),oklch(12%_0.01_60)_70%)]"
		>
			<div class="absolute inset-x-0 top-[22%] mx-auto w-[58%] rotate-[-3deg]">
				<ArtTile variant={0} framed class="aspect-[4/5] rounded-sm" />
			</div>
			<svg
				viewBox="0 0 100 64"
				class="absolute top-[18%] left-1/2 w-[74%] -translate-x-1/2 opacity-90 motion-safe:animate-pulse"
				aria-hidden="true"
			>
				<g fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round">
					<path d="M8 18 L8 8 L28 8" /><path d="M92 18 L92 8 L72 8" /><path
						d="M8 46 L8 56 L28 56"
					/><path d="M92 46 L92 56 L72 56" />
				</g>
			</svg>
			<div
				class="absolute inset-x-3 bottom-3 rounded-2xl border border-white/15 bg-[oklch(16%_0.01_280)]/70 p-3.5 text-white backdrop-blur-md"
			>
				<p class="font-display text-base leading-tight font-semibold">Harbor Light</p>
				<p class="mt-0.5 text-[11px] text-white/75">Ines Marlow, 2019</p>
				<p class="text-[11px] text-white/60">Oil on panel · 40 × 30 cm</p>
				<p class="mt-2 text-[11px] leading-4 text-white/70">
					Bought the week we moved in. The light in it is the light of the old apartment.
				</p>
				<div class="mt-3 grid grid-cols-6 gap-1">
					{#each emojis as emoji (emoji)}
						<span class="grid h-7 place-items-center rounded-lg bg-white/10 text-sm">{emoji}</span>
					{/each}
				</div>
				<p class="mt-2 text-[10px] text-white/50">12 reactions from guests</p>
			</div>
		</div>
	</div>
</div>
