<script lang="ts">
	import { Camera, Fingerprint, ScanLine, Sparkles } from 'lucide-svelte';

	const steps = [
		{
			icon: Camera,
			title: 'Photograph your pieces',
			body: 'Add a photo of each work as you catalog it. One phone picture is enough to start.'
		},
		{
			icon: Fingerprint,
			title: 'It learns the piece on the spot',
			body: 'Your browser finds the artwork in the photo and turns it into a compact visual fingerprint. Nothing to train, nothing to wait for.'
		},
		{
			icon: Sparkles,
			title: 'It keeps getting better',
			body: 'Every time the scanner recognizes a piece on your wall it remembers that view, so your real lighting and angles count too.'
		},
		{
			icon: ScanLine,
			title: 'Point a phone at a wall',
			body: 'Two small models run entirely in the browser: one spots artworks, one tells yours apart. Guests see the story and react; no app install, no frames leave the device.'
		}
	];
</script>

<section id="how" class="scroll-mt-24 bg-paper-deep/50 py-16 sm:py-24" aria-labelledby="how-title">
	<div class="mx-auto max-w-6xl px-5 sm:px-8">
		<div class="max-w-2xl">
			<p class="text-xs font-semibold tracking-[0.18em] text-accent-deep uppercase">
				How recognition works
			</p>
			<h2
				id="how-title"
				class="mt-3 font-display text-3xl font-medium tracking-tight text-balance sm:text-4xl"
			>
				Your photos are all it needs to know your art.
			</h2>
			<p class="mt-4 text-base leading-7 text-ink-soft">
				No cloud vision API, no sending images of your home to a third party, and no training step.
				A general artwork detector and a vision foundation model (built with DINOv3) run on the
				phone and compare what they see with fingerprints of your own photos.
			</p>
		</div>
		<ol class="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
			{#each steps as step, index (step.title)}
				<li
					class="relative flex flex-col gap-3 rounded-card border border-line bg-card p-6 shadow-card"
				>
					<div class="flex items-center justify-between">
						<span class="grid size-10 place-items-center rounded-xl bg-accent-soft text-accent-deep"
							><step.icon class="size-5" aria-hidden="true" /></span
						>
						<span class="font-display text-2xl text-ink-mute">0{index + 1}</span>
					</div>
					<h3 class="font-medium text-ink">{step.title}</h3>
					<p class="text-sm leading-6 text-ink-soft">{step.body}</p>
				</li>
			{/each}
		</ol>
	</div>
</section>
