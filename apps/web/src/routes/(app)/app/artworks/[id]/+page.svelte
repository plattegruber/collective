<script lang="ts">
	import { enhance } from '$app/forms';
	import { invalidate } from '$app/navigation';
	import { Eye, EyeOff, Pencil, ScanLine, Trash2 } from 'lucide-svelte';
	import { Badge, Button, Card } from '#lib/components/index.js';
	import ImageGallery from '#lib/components/ImageGallery.svelte';
	import { STATUS_LABELS } from '#lib/schemas/artwork.js';
	import { formatDimensions } from '#lib/utils/dimensions.js';
	import { byline, formatDate } from '#lib/utils/format.js';
	import { formatMoney } from '#lib/utils/money.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { formatLoadStage, loadRecognizer } from '#lib/recognition/loader.js';
	import { loadImage, teachPhoto } from '#lib/recognition/teach.js';
	import { captureToFile, clearCapture, peekCapture } from '#lib/scanner/capture.js';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();

	let teachStatus = $state('');
	let teachingCount = $state(0);
	const pendingImages = $derived(
		data.recognizer
			? data.artwork.images.filter((image) => !data.recognizer!.indexedImageIds.includes(image.id))
			: []
	);

	/** Index one photo in the browser: find the piece, embed it, store the prototype. */
	async function teach(imageId: string, url: string) {
		if (!data.recognizer) return;
		teachingCount += 1;
		try {
			const models = await loadRecognizer(
				data.recognizer.manifest,
				(stage) => (teachStatus = formatLoadStage(stage))
			);
			teachStatus = 'Learning this photo';
			await teachPhoto(models, data.artwork.id, imageId, await loadImage(url));
		} catch (cause) {
			toasts.push(cause instanceof Error ? cause.message : 'Could not index the photo.', 'error');
		} finally {
			teachingCount -= 1;
			if (teachingCount === 0) {
				teachStatus = '';
				void invalidate('app:artwork');
			}
		}
	}

	async function teachPending() {
		for (const image of pendingImages) await teach(image.id, `/media/${image.thumbKey}`);
	}

	let gallery: ReturnType<typeof ImageGallery> | undefined = $state();

	/** A piece captured by the scanner opens in the studio as this artwork's first photo. */
	async function attachCapture() {
		const capture = peekCapture();
		if (!capture || !gallery) return;
		clearCapture();
		try {
			gallery.openStudio([await captureToFile(capture)]);
		} catch (cause) {
			toasts.push(cause instanceof Error ? cause.message : 'Could not attach the photo.', 'error');
		}
	}

	$effect(() => {
		if (data.created && gallery) void attachCapture();
	});

	const artwork = $derived(data.artwork);
	const dims = $derived(formatDimensions(artwork, artwork.dimensionUnit));
	const sub = $derived(byline(artwork.artist, artwork.year, artwork.dateText));

	let confirmingDelete = $state(false);

	$effect(() => {
		if (form?.message) toasts.push(form.message, 'error');
	});

	const facts = $derived(
		[
			['Medium', artwork.medium],
			['Dimensions', dims],
			['Location', artwork.location],
			[
				'Acquired',
				[formatDate(artwork.acquiredOn), artwork.acquiredFrom].filter(Boolean).join(' · ')
			],
			[
				'Purchase price',
				artwork.purchasePrice === null ? '' : formatMoney(artwork.purchasePrice, artwork.currency)
			],
			[
				'Estimated value',
				artwork.estimatedValue === null ? '' : formatMoney(artwork.estimatedValue, artwork.currency)
			]
		].filter(([, value]) => Boolean(value)) as [string, string][]
	);
</script>

<svelte:head><title>{artwork.title} · Collective</title></svelte:head>

<div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
	<div class="min-w-0">
		<a
			href="/app/artworks"
			class="text-xs font-semibold tracking-[0.18em] text-ink-mute uppercase hover:text-ink"
			>← Artworks</a
		>
		<h1 class="mt-1 font-display text-3xl font-medium tracking-tight text-balance sm:text-4xl">
			{artwork.title}
		</h1>
		{#if sub}<p class="mt-1 text-base text-ink-soft">{sub}</p>{/if}
		<div class="mt-3 flex flex-wrap items-center gap-2">
			<Badge tone={artwork.status === 'owned' ? 'success' : 'neutral'}
				>{STATUS_LABELS[artwork.status]}</Badge
			>
			{#if artwork.isPublic}<Badge tone="accent">Public</Badge>{/if}
			{#each artwork.tags as tag (tag.id)}
				<a
					href={`/app/artworks?tag=${encodeURIComponent(tag.name)}`}
					class="rounded-full border border-line px-2.5 py-0.5 text-xs text-ink-soft hover:border-ink/40 hover:text-ink"
					>{tag.name}</a
				>
			{/each}
		</div>
	</div>
	<div class="flex shrink-0 flex-wrap gap-2">
		<Button href={`/app/artworks/${artwork.id}/edit`} variant="secondary"
			><Pencil class="size-4" aria-hidden="true" /> Edit</Button
		>
		<form method="POST" action="?/setPublic" use:enhance>
			<input type="hidden" name="isPublic" value={String(!artwork.isPublic)} />
			<Button
				type="submit"
				variant="ghost"
				title={artwork.isPublic ? 'Hide from public gallery' : 'Show in public gallery'}
			>
				{#if artwork.isPublic}<EyeOff class="size-4" aria-hidden="true" /> Make private{:else}<Eye
						class="size-4"
						aria-hidden="true"
					/> Make public{/if}
			</Button>
		</form>
	</div>
</div>

{#if data.created}
	<p class="mt-6 rounded-xl bg-success-soft px-4 py-3 text-sm text-success">
		Saved. Add a few photos so the piece is recognizable later.
	</p>
{/if}

<div class="mt-8 grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
	<section aria-label="Photos">
		<ImageGallery
			bind:this={gallery}
			artworkId={artwork.id}
			images={artwork.images}
			primaryImageId={artwork.primaryImageId}
			manifest={data.recognizer?.manifest ?? null}
			recordedAspect={artwork.widthCm && artwork.heightCm
				? artwork.widthCm / artwork.heightCm
				: null}
			onchange={() => invalidate('app:artwork')}
			onuploaded={(image) => teach(image.id, image.thumbUrl)}
		/>
	</section>

	<div class="flex flex-col gap-6">
		{#if artwork.description}
			<Card>
				<h2 class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">About</h2>
				<p class="mt-2 text-sm leading-6 whitespace-pre-line text-ink">{artwork.description}</p>
			</Card>
		{/if}
		{#if facts.length > 0}
			<Card>
				<dl class="divide-y divide-line">
					{#each facts as [label, value] (label)}
						<div class="grid grid-cols-[8rem_1fr] gap-3 py-2.5 text-sm first:pt-0 last:pb-0">
							<dt class="text-ink-mute">{label}</dt>
							<dd class="text-ink">{value}</dd>
						</div>
					{/each}
				</dl>
			</Card>
		{/if}
		{#if artwork.provenance}
			<Card>
				<h2 class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">Provenance</h2>
				<p class="mt-2 text-sm leading-6 whitespace-pre-line">{artwork.provenance}</p>
			</Card>
		{/if}
		{#if artwork.notes}
			<Card class="bg-paper-deep/50">
				<h2 class="text-xs font-semibold tracking-[0.16em] text-ink-mute uppercase">
					Private notes
				</h2>
				<p class="mt-2 text-sm leading-6 whitespace-pre-line">{artwork.notes}</p>
			</Card>
		{/if}

		<Card class="flex flex-col gap-3">
			<div class="flex items-center gap-2">
				<ScanLine class="size-4 text-accent-deep" aria-hidden="true" />
				<h2 class="text-sm font-medium">Recognition</h2>
			</div>
			{#if !data.recognizer}
				<p class="text-sm text-ink-soft">
					The recognizer is not available on this installation yet.
				</p>
			{:else if artwork.images.length === 0}
				<p class="text-sm text-ink-soft">
					Add a photo and the scanner knows this piece straight away. Nothing to train.
				</p>
			{:else}
				<p class="text-sm text-ink-soft">
					{#if teachStatus}
						{teachStatus}…
					{:else if pendingImages.length === 0}
						Known from {artwork.images.length === 1
							? 'its photo'
							: `all ${artwork.images.length} photos`}{#if data.recognizer.memories > 0}
							and {data.recognizer.memories} view{data.recognizer.memories === 1 ? '' : 's'} learned while
							scanning{/if}.
					{:else}
						{pendingImages.length} photo{pendingImages.length === 1 ? '' : 's'} not indexed yet.
					{/if}
				</p>
				{#if pendingImages.length > 0 && !teachStatus}
					<div>
						<Button type="button" size="sm" variant="secondary" onclick={teachPending}
							>Index now</Button
						>
					</div>
				{/if}
				<p class="text-xs text-ink-mute">{data.recognizer.manifest.config.attribution}</p>
			{/if}
		</Card>

		<div class="rounded-card border border-danger/20 p-4">
			{#if confirmingDelete}
				<p class="text-sm">
					Delete “{artwork.title}” and all of its photos? This cannot be undone.
				</p>
				<form method="POST" action="?/delete" class="mt-3 flex gap-2" use:enhance>
					<Button type="submit" variant="danger" size="sm">Yes, delete</Button>
					<Button type="button" variant="ghost" size="sm" onclick={() => (confirmingDelete = false)}
						>Keep it</Button
					>
				</form>
			{:else}
				<Button
					type="button"
					variant="ghost"
					size="sm"
					class="text-danger hover:bg-danger-soft hover:text-danger"
					onclick={() => (confirmingDelete = true)}
				>
					<Trash2 class="size-4" aria-hidden="true" /> Delete artwork
				</Button>
			{/if}
		</div>
	</div>
</div>
