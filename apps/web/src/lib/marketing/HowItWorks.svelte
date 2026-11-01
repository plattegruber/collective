<script lang="ts">
	import { Camera, Cpu, CloudUpload, ScanLine } from 'lucide-svelte';

	const steps = [
		{
			icon: Camera,
			title: 'Photograph your pieces',
			body: 'Add a few straight-on photos of each work as you catalog it. Phone photos are fine.'
		},
		{
			icon: Cpu,
			title: 'Train a tiny detector',
			body: 'Run the open-source trainer on your laptop or in CI. It builds a synthetic dataset from your photos and trains a mobile-sized model.'
		},
		{
			icon: CloudUpload,
			title: 'Publish to your account',
			body: 'The trainer uploads the model with an API token. It lives with your collection, versioned, and you can roll back.'
		},
		{
			icon: ScanLine,
			title: 'Point a phone at a wall',
			body: 'The model runs entirely in the browser. Guests see the story and react; no app install, no frames leave the device.'
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
				Your photos become a model that only knows your art.
			</h2>
			<p class="mt-4 text-base leading-7 text-ink-soft">
				No cloud vision API, no sending images of your home to a third party. The recognizer is a
				small on-device network trained on the pictures you already took.
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
