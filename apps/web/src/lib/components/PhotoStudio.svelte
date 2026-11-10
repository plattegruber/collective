<script lang="ts">
	import { Check, ImagePlus, X } from 'lucide-svelte';
	import type { RecognizerManifest } from '#lib/domain/recognition.js';
	import type { Point, Quad } from '#lib/images/geometry.js';
	import { prepareImage, type PreparedImage } from '#lib/images/prepare.js';
	import {
		DEFAULT_STUDIO_SETTINGS,
		PREVIEW_EDGE,
		detectCorners,
		openShot,
		prepareRender,
		rasterToCanvas,
		renderStudio,
		type StudioRender,
		type StudioSettings,
		type StudioShot
	} from '#lib/images/studio.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import Button from './Button.svelte';

	type Props = {
		/** Photos to work through, one picture each; extra shots can be added per picture. */
		files: File[];
		manifest: RecognizerManifest | null;
		/** Width / height of the piece from its record, offered as an output aspect ratio. */
		recordedAspect?: number | null;
		onsave: (prepared: PreparedImage, file: File) => Promise<void> | void;
		onclose: () => void;
	};

	let { files, manifest, recordedAspect = null, onsave, onclose }: Props = $props();

	let index = $state(0);
	let shots = $state.raw<StudioShot[]>([]);
	let active = $state(0);
	let settings = $state<StudioSettings>({ ...DEFAULT_STUDIO_SETTINGS });
	let useRecordedAspect = $state(false);
	let view = $state<'result' | 'corners'>('result');
	let preview = $state.raw<StudioRender | null>(null);
	let busy = $state<string | null>(null);
	let saving = $state(false);
	let comparing = $state(false);
	let dragging = $state<number | null>(null);
	let nextShotId = 1;
	let generation = 0;

	let resultCanvas: HTMLCanvasElement | undefined = $state();
	let sourceCanvas: HTMLCanvasElement | undefined = $state();
	let overlay: SVGSVGElement | undefined = $state();
	let addInput: HTMLInputElement | undefined = $state();

	// Display-sized copies of each shot's pixels, drawn once per shot (not reactive on purpose:
	// filled when a shot arrives, only read while rendering).
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	const sourceCache = new Map<number, { canvas: HTMLCanvasElement; thumb: string }>();
	const current = $derived(files[index] ?? null);
	const lead = $derived(shots[0] ?? null);
	const activeShot = $derived(shots[active] ?? lead);
	const effective = $derived<StudioSettings>({
		...settings,
		aspect: useRecordedAspect ? recordedAspect : null
	});

	function prime(shot: StudioShot): StudioShot {
		const full = rasterToCanvas(shot.raster);
		const scale = Math.min(1, PREVIEW_EDGE / Math.max(full.width, full.height));
		const canvas = document.createElement('canvas');
		canvas.width = Math.round(full.width * scale);
		canvas.height = Math.round(full.height * scale);
		const context = canvas.getContext('2d');
		if (context) {
			context.imageSmoothingQuality = 'high';
			context.drawImage(full, 0, 0, canvas.width, canvas.height);
		}
		const thumbCanvas = document.createElement('canvas');
		const thumbScale = 112 / Math.max(canvas.width, canvas.height);
		thumbCanvas.width = Math.max(1, Math.round(canvas.width * thumbScale));
		thumbCanvas.height = Math.max(1, Math.round(canvas.height * thumbScale));
		thumbCanvas.getContext('2d')?.drawImage(canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
		sourceCache.set(shot.id, { canvas, thumb: thumbCanvas.toDataURL('image/jpeg', 0.7) });
		return shot;
	}

	const sourceFor = (shot: StudioShot) => sourceCache.get(shot.id);

	async function start(file: File) {
		const mine = ++generation;
		shots = [];
		preview = null;
		active = 0;
		view = 'result';
		busy = 'Finding the piece';
		try {
			const shot = prime(await openShot(0, file, manifest, settings.prefer));
			if (mine !== generation) return;
			shots = [shot];
			await refresh();
		} catch (cause) {
			if (mine !== generation) return;
			busy = null;
			toasts.push(cause instanceof Error ? cause.message : 'Could not open the photo.', 'error');
			await skip();
		}
	}

	async function refresh() {
		const mine = ++generation;
		if (shots.length === 0) return;
		busy = shots.length > 1 ? 'Removing reflections' : 'Rendering';
		try {
			const render = await renderStudio(shots, effective, PREVIEW_EDGE);
			if (mine !== generation) return;
			preview = render;
		} catch (cause) {
			if (mine !== generation) return;
			toasts.push(cause instanceof Error ? cause.message : 'Could not render the photo.', 'error');
		} finally {
			if (mine === generation) busy = null;
		}
	}

	$effect(() => {
		if (current) void start(current);
	});

	$effect(() => {
		const canvas = resultCanvas;
		const render = preview;
		if (!canvas || !render) return;
		canvas.width = render.raster.width;
		canvas.height = render.raster.height;
		canvas
			.getContext('2d')
			?.putImageData(
				new ImageData(render.raster.data, render.raster.width, render.raster.height),
				0,
				0
			);
	});

	$effect(() => {
		const canvas = sourceCanvas;
		const shot = activeShot;
		if (!canvas || !shot) return;
		const source = sourceFor(shot)?.canvas;
		if (!source) return;
		canvas.width = source.width;
		canvas.height = source.height;
		canvas.getContext('2d')?.drawImage(source, 0, 0);
	});

	function setSetting<K extends keyof StudioSettings>(key: K, value: StudioSettings[K]) {
		settings = { ...settings, [key]: value };
		if (key === 'prefer') {
			// Re-detect corners the collector has not touched.
			shots = shots.map((shot) =>
				shot.confidence > 0 || !shot.detected
					? {
							...shot,
							...pick(detectCorners(shot.raster, shot.box, shot.detected, settings.prefer))
						}
					: shot
			);
		}
		void refresh();
	}

	function pick(detection: { quad: Quad; confidence: number }) {
		return { quad: detection.quad, confidence: detection.confidence };
	}

	function toggleRecordedAspect() {
		useRecordedAspect = !useRecordedAspect;
		void refresh();
	}

	async function addShots(list: FileList | null) {
		if (!list || list.length === 0 || !lead) return;
		busy = 'Reading the extra shots';
		const added: StudioShot[] = [];
		for (const file of Array.from(list)) {
			try {
				added.push(prime(await openShot(nextShotId++, file, manifest, settings.prefer, lead)));
			} catch (cause) {
				toasts.push(
					cause instanceof Error ? cause.message : `Could not read ${file.name}.`,
					'error'
				);
			}
		}
		shots = [...shots, ...added];
		await refresh();
	}

	function removeShot(id: number) {
		if (shots.length <= 1) return;
		shots = shots.filter((shot) => shot.id !== id);
		sourceCache.delete(id);
		active = Math.min(active, shots.length - 1);
		void refresh();
	}

	/** Pointer position in the active shot's pixel space. */
	function toImage(event: PointerEvent): Point | null {
		const shot = activeShot;
		if (!overlay || !shot) return null;
		const rect = overlay.getBoundingClientRect();
		return {
			x: Math.min(
				shot.raster.width,
				Math.max(0, ((event.clientX - rect.left) / rect.width) * shot.raster.width)
			),
			y: Math.min(
				shot.raster.height,
				Math.max(0, ((event.clientY - rect.top) / rect.height) * shot.raster.height)
			)
		};
	}

	function startDrag(event: PointerEvent, corner: number) {
		(event.currentTarget as Element).setPointerCapture(event.pointerId);
		dragging = corner;
	}

	function drag(event: PointerEvent) {
		if (dragging === null) return;
		const point = toImage(event);
		const shot = activeShot;
		if (!point || !shot) return;
		const quad = shot.quad.map((p, i) => (i === dragging ? point : p)) as Quad;
		shots = shots.map((s) => (s.id === shot.id ? { ...s, quad, confidence: 0 } : s));
	}

	function endDrag() {
		if (dragging === null) return;
		dragging = null;
		void refresh();
	}

	async function save() {
		if (!lead || saving) return;
		saving = true;
		busy = 'Rendering at full size';
		try {
			const render = await renderStudio(shots, effective);
			const prepared = await prepareRender(render, lead.file);
			await onsave(prepared, lead.file);
			await advance();
		} catch (cause) {
			toasts.push(cause instanceof Error ? cause.message : 'Could not save the photo.', 'error');
		} finally {
			saving = false;
			busy = null;
		}
	}

	/** Uploads the photo exactly as it came in. */
	async function skip() {
		if (!current || saving) return;
		saving = true;
		try {
			await onsave(await prepareImage(current), current);
			await advance();
		} catch (cause) {
			toasts.push(cause instanceof Error ? cause.message : 'Could not upload the photo.', 'error');
		} finally {
			saving = false;
		}
	}

	async function advance() {
		sourceCache.clear();
		if (index + 1 < files.length) {
			index += 1;
		} else {
			onclose();
		}
	}

	function onKey(event: KeyboardEvent) {
		if (event.key === 'Escape') onclose();
	}

	const cornerHint = $derived(
		!lead
			? ''
			: lead.confidence === 0 && lead.detected && view !== 'corners'
				? 'Edges were unclear. Check the corners.'
				: lead.confidence === 0 && !lead.detected
					? 'Using the whole photo. Drag the corners if the piece is smaller.'
					: ''
	);
	const points = $derived(activeShot?.quad.map((p) => `${p.x},${p.y}`).join(' ') ?? '');
	const chip =
		'inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 transition focus-visible:focus-ring';
	const chipOff = 'border-white/20 text-paper/80 hover:border-white/50 hover:text-paper';
	const chipOn = 'border-white bg-white text-ink';
	const handleRadius = $derived(
		activeShot ? Math.max(activeShot.raster.width, activeShot.raster.height) * 0.014 : 10
	);
</script>

<svelte:window onkeydown={onKey} />

<div
	class="fixed inset-0 z-50 flex flex-col bg-ink text-paper"
	role="dialog"
	aria-modal="true"
	aria-label="Photo studio"
>
	<header class="flex items-center justify-between gap-3 px-4 py-3">
		<div class="min-w-0">
			<h2 class="font-display text-lg font-medium">Photo studio</h2>
			<p class="truncate text-xs text-paper/60">
				{#if files.length > 1}Photo {index + 1} of {files.length} ·
				{/if}{current?.name ?? ''}
			</p>
		</div>
		<button
			type="button"
			class="grid size-9 place-items-center rounded-full text-paper/80 hover:bg-white/10 hover:text-paper focus-visible:focus-ring"
			onclick={onclose}
			aria-label="Close"><X class="size-5" aria-hidden="true" /></button
		>
	</header>

	<div class="relative flex min-h-0 flex-1 items-center justify-center px-3">
		{#if view === 'corners' && activeShot}
			<div
				class="relative max-h-full max-w-full"
				style={`aspect-ratio:${activeShot.raster.width} / ${activeShot.raster.height}; height:min(100%, calc((100vw - 1.5rem) * ${activeShot.raster.height / activeShot.raster.width}))`}
			>
				<canvas bind:this={sourceCanvas} class="block size-full rounded-lg"></canvas>
				<svg
					bind:this={overlay}
					class="absolute inset-0 size-full touch-none"
					viewBox={`0 0 ${activeShot.raster.width} ${activeShot.raster.height}`}
					preserveAspectRatio="none"
					onpointermove={drag}
					onpointerup={endDrag}
					onpointercancel={endDrag}
					role="application"
					aria-label="Drag the corners onto the piece"
				>
					<polygon
						{points}
						fill="rgba(255,255,255,0.08)"
						stroke="#fff"
						stroke-width="2"
						vector-effect="non-scaling-stroke"
					/>
					{#each activeShot.quad as corner, i (i)}
						<circle
							cx={corner.x}
							cy={corner.y}
							r={handleRadius * 2.2}
							fill="transparent"
							class="cursor-move"
							onpointerdown={(event) => startDrag(event, i)}
							role="presentation"
						/>
						<circle
							cx={corner.x}
							cy={corner.y}
							r={handleRadius}
							fill={dragging === i ? '#fff' : 'rgba(255,255,255,0.9)'}
							stroke="#111"
							stroke-width="2"
							vector-effect="non-scaling-stroke"
							class="pointer-events-none"
						/>
					{/each}
				</svg>
			</div>
		{:else}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="relative flex max-h-full max-w-full items-center justify-center"
				onpointerdown={() => (comparing = true)}
				onpointerup={() => (comparing = false)}
				onpointerleave={() => (comparing = false)}
			>
				<canvas
					bind:this={resultCanvas}
					class={['block max-h-full max-w-full rounded-lg object-contain', comparing && 'hidden']}
					style="max-height: calc(100vh - 15rem)"
				></canvas>
				{#if comparing && lead}
					<canvas
						bind:this={sourceCanvas}
						class="block max-h-full max-w-full rounded-lg object-contain"
						style="max-height: calc(100vh - 15rem)"
					></canvas>
				{/if}
				{#if preview && !comparing}
					<span
						class="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-black/50 px-2.5 py-1 text-[11px] text-white/80"
						>Hold to compare</span
					>
				{/if}
			</div>
		{/if}
		{#if busy}
			<div
				class="absolute inset-x-0 bottom-3 mx-auto flex w-max items-center gap-2 rounded-full bg-black/60 px-3 py-1.5 text-xs"
				aria-live="polite"
			>
				<span
					class="size-3 animate-spin rounded-full border-2 border-current border-t-transparent"
					aria-hidden="true"
				></span>
				{busy}…
			</div>
		{/if}
	</div>

	<div class="flex flex-col gap-3 px-4 pt-3 pb-[max(1rem,env(safe-area-inset-bottom))]">
		{#if shots.length > 1 || view === 'corners'}
			<div class="flex items-center gap-2 overflow-x-auto">
				{#each shots as shot, i (shot.id)}
					<div class="relative shrink-0">
						<button
							type="button"
							class={[
								'block size-14 overflow-hidden rounded-lg border-2 bg-black',
								i === active ? 'border-white' : 'border-white/20'
							]}
							onclick={() => (active = i)}
							aria-label={`Shot ${i + 1}`}
							aria-pressed={i === active}
						>
							<img src={sourceFor(shot)?.thumb} alt="" class="size-full object-cover" />
						</button>
						{#if shots.length > 1}
							<button
								type="button"
								class="absolute -top-1.5 -right-1.5 grid size-5 place-items-center rounded-full bg-white text-ink"
								onclick={() => removeShot(shot.id)}
								aria-label={`Remove shot ${i + 1}`}><X class="size-3" aria-hidden="true" /></button
							>
						{/if}
					</div>
				{/each}
			</div>
		{/if}

		<div class="flex flex-wrap items-center gap-2 text-sm">
			<button
				type="button"
				class={[chip, settings.straighten ? chipOn : chipOff]}
				aria-pressed={settings.straighten}
				onclick={() => setSetting('straighten', !settings.straighten)}>Straighten</button
			>
			<button
				type="button"
				class={[chip, settings.colour ? chipOn : chipOff]}
				aria-pressed={settings.colour}
				onclick={() => setSetting('colour', !settings.colour)}>Auto colour</button
			>
			<div
				class="inline-flex overflow-hidden rounded-full border border-white/20"
				role="group"
				aria-label="Crop"
			>
				<button
					type="button"
					class={[
						'px-3 py-1.5',
						settings.prefer === 'outer' ? 'bg-white text-ink' : 'text-paper/80'
					]}
					aria-pressed={settings.prefer === 'outer'}
					onclick={() => setSetting('prefer', 'outer')}>Keep frame</button
				>
				<button
					type="button"
					class={[
						'px-3 py-1.5',
						settings.prefer === 'inner' ? 'bg-white text-ink' : 'text-paper/80'
					]}
					aria-pressed={settings.prefer === 'inner'}
					onclick={() => setSetting('prefer', 'inner')}>Art only</button
				>
			</div>
			{#if recordedAspect}
				<button
					type="button"
					class={[chip, useRecordedAspect ? chipOn : chipOff]}
					aria-pressed={useRecordedAspect}
					onclick={toggleRecordedAspect}>Recorded proportions</button
				>
			{/if}
			<button
				type="button"
				class={[chip, view === 'corners' ? chipOn : chipOff]}
				aria-pressed={view === 'corners'}
				onclick={() => (view = view === 'corners' ? 'result' : 'corners')}
				>{view === 'corners' ? 'Done with corners' : 'Adjust corners'}</button
			>
			<button type="button" class={[chip, chipOff]} onclick={() => addInput?.click()}>
				<ImagePlus class="size-4" aria-hidden="true" />
				{shots.length > 1 ? 'Add shot' : 'Remove reflections'}
			</button>
			<input
				bind:this={addInput}
				type="file"
				accept="image/*"
				multiple
				class="sr-only"
				onchange={(event) => {
					const input = event.currentTarget;
					void addShots(input.files);
					input.value = '';
				}}
			/>
		</div>

		<p class="min-h-4 text-xs text-paper/60">
			{#if preview?.fused}
				{shots.length} shots fused · reflections replaced on {Math.round(
					preview.fused.changed * 100
				)}% of the picture
			{:else if shots.length === 1 && cornerHint}
				{cornerHint}
			{:else if shots.length === 1}
				Reflections in the glass? Take two more shots from a step to each side and add them.
			{/if}
		</p>

		<div class="flex items-center justify-between gap-3">
			<Button
				variant="ghost"
				class="text-paper/80 hover:bg-white/10 hover:text-paper"
				onclick={skip}
				disabled={saving}>Use as shot</Button
			>
			<Button
				variant="inverse"
				onclick={save}
				loading={saving}
				disabled={!preview || Boolean(busy)}
			>
				<Check class="size-4" aria-hidden="true" />
				{index + 1 < files.length ? 'Save & next' : 'Save photo'}
			</Button>
		</div>
	</div>
</div>
