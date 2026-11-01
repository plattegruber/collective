<script lang="ts">
	const faqs = [
		{
			q: 'Who sees my collection?',
			a: 'Only you, unless you turn on your public gallery and mark individual pieces public. Prices, values, provenance, and private notes never appear on public pages.'
		},
		{
			q: 'Does the scanner upload video from my camera?',
			a: 'No. The detector runs in the browser on the phone. Camera frames are processed locally and discarded; nothing is recorded or sent anywhere.'
		},
		{
			q: 'Where are my photos stored?',
			a: 'Privately, in object storage tied to your account, served only to you or through your public gallery when you choose. Images are resized on your device before upload.'
		},
		{
			q: 'Can I get my data out?',
			a: 'Yes. Your collection is exportable at any time, and the training pipeline uses the same open API with a token you create and can revoke.'
		},
		{
			q: 'How much does it cost?',
			a: 'Collective is free during early access for personal collections.'
		},
		{
			q: 'Do I need to train a model to use it?',
			a: 'No. The inventory works on its own. Recognition is an optional feature you can set up whenever you have a few photos per piece.'
		}
	];
</script>

<section id="faq" class="scroll-mt-24 py-16 sm:py-24" aria-labelledby="faq-title">
	<div class="mx-auto max-w-3xl px-5 sm:px-8">
		<p class="text-xs font-semibold tracking-[0.18em] text-accent-deep uppercase">Questions</p>
		<h2 id="faq-title" class="mt-3 font-display text-3xl font-medium tracking-tight sm:text-4xl">
			Good to know
		</h2>
		<div class="mt-10 divide-y divide-line border-y border-line">
			{#each faqs as faq (faq.q)}
				<details class="group py-5">
					<summary
						class="flex cursor-pointer list-none items-center justify-between gap-4 rounded-md text-base font-medium text-ink focus-visible:focus-ring [&::-webkit-details-marker]:hidden"
					>
						{faq.q}
						<span
							class="grid size-7 shrink-0 place-items-center rounded-full border border-line text-ink-soft transition group-open:rotate-45"
							aria-hidden="true">+</span
						>
					</summary>
					<p class="mt-3 max-w-prose text-sm leading-7 text-ink-soft">{faq.a}</p>
				</details>
			{/each}
		</div>
	</div>
</section>
