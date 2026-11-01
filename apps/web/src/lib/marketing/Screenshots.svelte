<script lang="ts">
	import { screenshots, screenshotsAvailable } from './screenshots.js';
	import MockArtworkDetail from './MockArtworkDetail.svelte';
	import MockCollectionGrid from './MockCollectionGrid.svelte';
	import MockDashboard from './MockDashboard.svelte';
	import MockScanner from './MockScanner.svelte';
</script>

<section id="screens" class="scroll-mt-24 py-16 sm:py-24" aria-labelledby="screens-title">
	<div class="mx-auto max-w-6xl px-5 sm:px-8">
		<div class="max-w-2xl">
			<p class="text-xs font-semibold tracking-[0.18em] text-accent-deep uppercase">
				A look inside
			</p>
			<h2
				id="screens-title"
				class="mt-3 font-display text-3xl font-medium tracking-tight text-balance sm:text-4xl"
			>
				Built to be opened often.
			</h2>
			<p class="mt-4 text-base leading-7 text-ink-soft">
				Calm screens, quick entry, nothing in the way of the work itself.
			</p>
		</div>
		{#if screenshotsAvailable}
			<ul class="mt-12 grid gap-8 sm:grid-cols-2">
				{#each screenshots as shot (shot.src)}
					<li class="flex flex-col gap-3">
						<figure>
							<img
								src={shot.src}
								alt={shot.alt}
								width={shot.width}
								height={shot.height}
								loading="lazy"
								decoding="async"
								class="w-full rounded-card border border-line bg-paper-deep shadow-lift"
							/>
							<figcaption class="mt-3 text-sm font-medium text-ink-soft">{shot.caption}</figcaption>
						</figure>
					</li>
				{/each}
			</ul>
		{:else}
			<div class="mt-12 grid gap-8 lg:grid-cols-[1.3fr_1fr]">
				<div class="flex flex-col gap-8">
					<MockDashboard />
					<MockCollectionGrid />
				</div>
				<div class="flex flex-col gap-8">
					<MockArtworkDetail />
					<div class="hidden lg:block"><MockScanner /></div>
				</div>
			</div>
		{/if}
	</div>
</section>
