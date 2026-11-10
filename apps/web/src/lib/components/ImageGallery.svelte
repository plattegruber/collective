<script lang="ts">
	import { enhance } from '$app/forms';
	import { Download, ImagePlus, Star, Trash2, UploadCloud } from 'lucide-svelte';
	import type { RecognizerManifest } from '#lib/domain/recognition.js';
	import { uploadArtworkImage, type PreparedImage } from '#lib/images/prepare.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import type { ArtworkImage } from '#lib/server/db/schema.js';
	import Button from './Button.svelte';
	import PhotoStudio from './PhotoStudio.svelte';

	type Props = {
		artworkId: string;
		images: ArtworkImage[];
		primaryImageId: string | null;
		/** Lets the studio find the piece in a photo with the detector; null without a release. */
		manifest?: RecognizerManifest | null;
		/** Width / height from the record, offered by the studio as the output proportions. */
		recordedAspect?: number | null;
		onchange?: () => void;
		/** Fires once per successful upload, before `onchange`, with the stored image's urls. */
		onuploaded?: (image: { id: string; url: string; thumbUrl: string }) => void;
	};

	let {
		artworkId,
		images,
		primaryImageId,
		manifest = null,
		recordedAspect = null,
		onchange,
		onuploaded
	}: Props = $props();

	interface Pending {
		id: number;
		name: string;
		progress: number;
		preview: string;
		error?: string;
	}

	let pending = $state<Pending[]>([]);
	let dragging = $state(false);
	let selected = $state<string | null>(null);
	let fileInput: HTMLInputElement | undefined = $state();
	let nextId = 0;

	const ordered = $derived(
		[...images].sort((a, b) =>
			a.id === primaryImageId ? -1 : b.id === primaryImageId ? 1 : a.position - b.position
		)
	);
	const current = $derived(ordered.find((image) => image.id === selected) ?? ordered[0] ?? null);

	/** Photos waiting in the studio; the studio hands each back prepared, or as shot. */
	let studioFiles = $state<File[] | null>(null);
	let uploadedInStudio = 0;

	function handleFiles(files: FileList | File[]) {
		const list = Array.from(files).filter((file) => file.type.startsWith('image/'));
		if (list.length === 0) return;
		openStudio(list);
	}

	/** Opens the studio for `files`; the artwork page uses this for scanner captures. */
	export function openStudio(files: File[]) {
		uploadedInStudio = 0;
		studioFiles = files;
	}

	function closeStudio() {
		studioFiles = null;
		if (uploadedInStudio > 0) onchange?.();
	}

	async function uploadPrepared(prepared: PreparedImage, name: string) {
		const item: Pending = {
			id: nextId++,
			name,
			progress: 0,
			preview: URL.createObjectURL(prepared.thumb)
		};
		pending = [...pending, item];
		const update = (patch: Partial<Pending>) => {
			pending = pending.map((p) => (p.id === item.id ? { ...p, ...patch } : p));
		};
		try {
			const stored = await uploadArtworkImage(artworkId, prepared, (fraction) =>
				update({ progress: fraction })
			);
			pending = pending.filter((p) => p.id !== item.id);
			uploadedInStudio += 1;
			onuploaded?.(stored);
		} catch (cause) {
			const message = cause instanceof Error ? cause.message : 'Upload failed.';
			update({ error: message });
			toasts.push(`${name}: ${message}`, 'error');
			throw cause;
		} finally {
			URL.revokeObjectURL(item.preview);
		}
	}

	function onDrop(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		if (event.dataTransfer?.files) void handleFiles(event.dataTransfer.files);
	}
</script>

<div class="flex flex-col gap-4">
	{#if current}
		<figure class="overflow-hidden rounded-card border border-line bg-paper-deep shadow-card">
			<img
				src={`/media/${current.key}`}
				alt=""
				width={current.width}
				height={current.height}
				class="mx-auto max-h-[70vh] w-auto object-contain"
			/>
			{#if current.originalKey}
				<figcaption class="flex justify-end px-3 py-2 text-xs">
					<a
						href={`/media/${current.originalKey}`}
						download
						class="inline-flex items-center gap-1 text-ink-soft hover:text-ink"
						><Download class="size-3.5" aria-hidden="true" /> Original as shot</a
					>
				</figcaption>
			{/if}
		</figure>
	{/if}

	<div
		role="group"
		aria-label="Photos"
		class={[
			'flex flex-wrap gap-3 rounded-card border border-dashed p-3 transition',
			dragging ? 'border-accent bg-accent-soft/40' : 'border-line'
		].join(' ')}
		ondragover={(event) => {
			event.preventDefault();
			dragging = true;
		}}
		ondragleave={() => (dragging = false)}
		ondrop={onDrop}
	>
		{#each ordered as image (image.id)}
			<div class="group relative">
				<button
					type="button"
					class={[
						'block size-20 overflow-hidden rounded-xl border bg-paper-deep transition focus-visible:focus-ring',
						current?.id === image.id
							? 'border-ink ring-2 ring-ink/20'
							: 'border-line hover:border-ink/40'
					].join(' ')}
					onclick={() => (selected = image.id)}
					aria-label={`Show photo ${image.position + 1}`}
					aria-pressed={current?.id === image.id}
				>
					<img
						src={`/media/${image.thumbKey}`}
						alt=""
						class="size-full object-cover"
						loading="lazy"
					/>
				</button>
				{#if image.id === primaryImageId}
					<span
						class="absolute -top-1.5 -left-1.5 grid size-5 place-items-center rounded-full bg-ink text-paper"
						title="Cover photo"><Star class="size-3" aria-hidden="true" /></span
					>
				{/if}
				<div
					class="absolute inset-x-0 -bottom-1 flex justify-center gap-1 opacity-0 transition group-focus-within:opacity-100 group-hover:opacity-100"
				>
					{#if image.id !== primaryImageId}
						<form
							method="POST"
							action="?/setPrimary"
							use:enhance={() =>
								async ({ update }) => {
									await update();
									onchange?.();
								}}
						>
							<input type="hidden" name="imageId" value={image.id} />
							<button
								type="submit"
								class="grid size-6 place-items-center rounded-full bg-paper text-ink shadow-card hover:bg-white"
								title="Use as cover"
								><Star class="size-3" aria-hidden="true" /><span class="sr-only">Use as cover</span
								></button
							>
						</form>
					{/if}
					<form
						method="POST"
						action="?/deleteImage"
						use:enhance={() =>
							async ({ update }) => {
								await update();
								onchange?.();
							}}
					>
						<input type="hidden" name="imageId" value={image.id} />
						<button
							type="submit"
							class="grid size-6 place-items-center rounded-full bg-paper text-danger shadow-card hover:bg-white"
							title="Remove photo"
							><Trash2 class="size-3" aria-hidden="true" /><span class="sr-only">Remove photo</span
							></button
						>
					</form>
				</div>
			</div>
		{/each}

		{#each pending as item (item.id)}
			<div
				class="relative size-20 overflow-hidden rounded-xl border border-line bg-paper-deep"
				aria-live="polite"
			>
				<img src={item.preview} alt="" class="size-full object-cover opacity-60" />
				{#if item.error}
					<span
						class="absolute inset-0 grid place-items-center bg-danger/70 p-1 text-center text-[10px] font-medium text-white"
						>Failed</span
					>
				{:else}
					<span class="absolute inset-x-1 bottom-1 h-1 overflow-hidden rounded-full bg-white/50"
						><span
							class="block h-full bg-ink transition-[width]"
							style={`width:${Math.round(item.progress * 100)}%`}
						></span></span
					>
				{/if}
			</div>
		{/each}

		<button
			type="button"
			class="flex size-20 flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-line text-xs text-ink-soft transition hover:border-ink/40 hover:text-ink focus-visible:focus-ring"
			onclick={() => fileInput?.click()}
		>
			<ImagePlus class="size-5" aria-hidden="true" />
			Add
		</button>
		<input
			bind:this={fileInput}
			type="file"
			accept="image/jpeg,image/png,image/webp"
			multiple
			class="sr-only"
			onchange={(event) => {
				const input = event.currentTarget;
				if (input.files) void handleFiles(input.files);
				input.value = '';
			}}
		/>
	</div>

	{#if images.length === 0 && pending.length === 0}
		<div
			class="flex items-center gap-3 rounded-card bg-accent-soft/60 p-4 text-sm text-accent-deep"
		>
			<UploadCloud class="size-5 shrink-0" aria-hidden="true" />
			<p>
				Drop photos here or <Button
					type="button"
					variant="ghost"
					size="sm"
					class="-mx-1 h-auto px-1 text-accent-deep underline underline-offset-4"
					onclick={() => fileInput?.click()}>choose files</Button
				>. They are straightened and cropped for you; the originals are kept.
			</p>
		</div>
	{/if}
</div>

{#if studioFiles}
	<PhotoStudio
		files={studioFiles}
		{manifest}
		{recordedAspect}
		onsave={(prepared, file) => uploadPrepared(prepared, file.name)}
		onclose={closeStudio}
	/>
{/if}
