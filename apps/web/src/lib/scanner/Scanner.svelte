<script lang="ts">
	import { untrack } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { Camera, Plus, X } from 'lucide-svelte';
	import type { PrototypeIndexPayload, RecognizerManifest } from '#lib/domain/recognition.js';
	import { cropToBlob } from '#lib/recognition/frames.js';
	import {
		formatLoadStage,
		loadRecognizer,
		type RecognizerModels
	} from '#lib/recognition/loader.js';
	import { PrototypeIndex } from '#lib/recognition/matcher.js';
	import { Recognizer, type RecognitionFrame } from '#lib/recognition/recognizer.js';
	import { rebuildIndex, rememberScan } from '#lib/recognition/teach.js';
	import type { Box } from '#lib/recognition/types.js';
	import {
		CameraError,
		openRearCamera,
		queryCameraPermission,
		stopStream,
		type CameraPermission
	} from './camera.js';
	import { storeCapture } from './capture.js';
	import ScannerOverlay, { type ScannerArtwork } from './ScannerOverlay.svelte';
	import Button from '#lib/components/Button.svelte';

	type Props = {
		manifest: RecognizerManifest;
		index: PrototypeIndexPayload;
		artworks: Record<string, ScannerArtwork>;
		reactions?: boolean;
		backHref: string;
		title?: string;
		/** Owners teach the recognizer while scanning and can add unknown pieces. */
		canTeach?: boolean;
	};

	let {
		manifest,
		index,
		artworks,
		reactions = true,
		backHref,
		title = 'Scanner',
		canTeach = false
	}: Props = $props();

	type Phase = 'idle' | 'permission' | 'loading' | 'scanning' | 'error';
	let phase = $state<Phase>('idle');
	let status = $state('');
	let errorMessage = $state('');
	let permission = $state<CameraPermission>('unknown');
	let video: HTMLVideoElement | undefined = $state();
	let frame = $state<RecognitionFrame | null>(null);
	let fps = $state(0);
	let viewport = $state({ width: 1, height: 1 });
	let videoSize = $state({ width: 0, height: 0 });
	let backend = $state('');
	let adding = $state(false);
	/** `?debug` shows timings and similarities; handy when tuning thresholds on site. */
	const debug =
		typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('debug');

	let stream: MediaStream | null = null;
	let recognizer: Recognizer | null = null;
	let models: RecognizerModels | null = null;
	let running = false;
	/** Bumped by stop(); a start() that outlives its generation abandons its work. */
	let generation = 0;
	// The payload is fixed for the page; a new release or gallery means a new page load.
	const prototypeIndex = PrototypeIndex.fromPayload(untrack(() => index));
	/** Plain record, not reactive: nothing renders from it. */
	const remembered: Record<string, number> = {};
	let stableSince: { artworkId: string; since: number } | null = null;

	const currentId = $derived(frame?.current?.artworkId ?? null);
	const current = $derived(currentId ? (artworks[currentId] ?? null) : null);
	const unknown = $derived(canTeach && !currentId ? (frame?.unknown ?? null) : null);
	const prompts = [
		'Point me toward the art.',
		'Frame the whole piece.',
		'A little closer.',
		'Hold steady for a moment.'
	];
	let promptIndex = $state(0);
	let promptTimer: ReturnType<typeof setInterval> | undefined;

	/** The video is object-cover; map frame pixels onto the viewport. */
	function toScreen(box: Box): Box {
		const { width: vw, height: vh } = videoSize;
		if (!vw || !vh) return box;
		const scale = Math.max(viewport.width / vw, viewport.height / vh);
		const ox = (viewport.width - vw * scale) / 2;
		const oy = (viewport.height - vh * scale) / 2;
		return {
			x1: ox + box.x1 * scale,
			y1: oy + box.y1 * scale,
			x2: ox + box.x2 * scale,
			y2: oy + box.y2 * scale
		};
	}

	async function start() {
		const gen = ++generation;
		const cancelled = () => gen !== generation;
		errorMessage = '';
		phase = 'loading';
		try {
			status = 'Opening camera';
			const opened = await openRearCamera();
			if (cancelled()) {
				stopStream(opened);
				return;
			}
			stream = opened;
			if (!video) throw new Error('Video element missing.');
			video.srcObject = stream;
			await video.play();
			if (cancelled()) return;
			videoSize = { width: video.videoWidth, height: video.videoHeight };

			if (!models) {
				const loaded = await loadRecognizer(manifest, (stage) => (status = formatLoadStage(stage)));
				if (cancelled()) return;
				models = loaded;
				backend = loaded.backend;
			}
			recognizer = new Recognizer(models, prototypeIndex);
			phase = 'scanning';
			running = true;
			promptTimer = setInterval(() => (promptIndex = (promptIndex + 1) % prompts.length), 4000);
			void loop(gen);
			if (canTeach) void selfHeal(gen);
		} catch (cause) {
			if (cancelled()) return;
			stopStream(stream);
			stream = null;
			if (cause instanceof CameraError && cause.kind === 'denied') {
				permission = 'denied';
				phase = 'permission';
			} else {
				errorMessage = cause instanceof Error ? cause.message : 'Something went wrong.';
				phase = 'error';
			}
		}
	}

	/** Photos uploaded before this release (or from a device without the models) get indexed now. */
	async function selfHeal(gen: number) {
		try {
			await rebuildIndex(undefined, undefined, (artworkId, embedding) => {
				if (gen === generation) prototypeIndex.add(artworkId, embedding);
			});
		} catch (cause) {
			console.warn('Index self-heal failed', cause);
		}
	}

	async function loop(gen: number) {
		let lastTick = performance.now();
		let frames = 0;
		while (running && gen === generation && recognizer && video) {
			try {
				const result = await recognizer.step(video);
				if (gen !== generation) return;
				if (result) {
					frame = result;
					void maybeRemember(result);
				}
			} catch (cause) {
				console.error('Recognition failed', cause);
			}
			frames += 1;
			const now = performance.now();
			if (now - lastTick >= 1000) {
				fps = frames;
				frames = 0;
				lastTick = now;
			}
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
		}
	}

	/** After a match has held for a moment, keep that view as a memory so it is found faster next time. */
	async function maybeRemember(result: RecognitionFrame) {
		if (!canTeach || !models) return;
		const embedding = result.current?.region?.embedding;
		if (!result.current || !embedding) return;
		const { artworkId } = result.current;
		const now = performance.now();
		if (stableSince?.artworkId !== artworkId) {
			stableSince = { artworkId, since: now };
			return;
		}
		if (now - stableSince.since < 2500) return;
		const last = remembered[artworkId] ?? -Infinity;
		if (now - last < 60_000) return;
		remembered[artworkId] = now;
		try {
			await rememberScan(models, artworkId, embedding);
			prototypeIndex.add(artworkId, embedding);
		} catch (cause) {
			console.warn('Could not store scan memory', cause);
		}
	}

	async function addUnknown() {
		if (!video || !unknown || adding) return;
		adding = true;
		try {
			await storeCapture(await cropToBlob(video, unknown.box));
			await goto(resolve('/(app)/app/artworks/new'));
		} catch (cause) {
			console.error('Could not capture the piece', cause);
			adding = false;
		}
	}

	function stop() {
		generation += 1;
		running = false;
		clearInterval(promptTimer);
		stopStream(stream);
		stream = null;
		recognizer = null;
		frame = null;
		phase = 'idle';
	}

	// $effect only runs in the browser, so none of this touches the camera or ORT during SSR.
	$effect(() => {
		let cancelled = false;
		const measure = () => (viewport = { width: window.innerWidth, height: window.innerHeight });
		measure();
		window.addEventListener('resize', measure);
		void queryCameraPermission().then((state) => {
			if (cancelled) return;
			permission = state;
			if (state === 'granted') void start();
			else if (state === 'unsupported') {
				errorMessage =
					'This browser cannot access the camera. Try Safari on iOS or Chrome on Android.';
				phase = 'error';
			} else phase = 'permission';
		});
		return () => {
			cancelled = true;
			window.removeEventListener('resize', measure);
			stop();
		};
	});

	const debugLine = $derived.by(() => {
		if (!frame) return 'no frame';
		const best = [...frame.regions]
			.filter((r) => r.embedding)
			.sort((a, b) => b.similarity - a.similarity)[0];
		return `${backend} · detect ${frame.timings.detectMs.toFixed(0)}ms · embed ${frame.timings.embedMs.toFixed(0)}ms · ${frame.regions.length} region${frame.regions.length === 1 ? '' : 's'}${best ? ` · best ${best.artworkId ? (artworks[best.artworkId]?.title ?? best.artworkId) : 'unknown'} ${best.similarity.toFixed(2)}` : ''} · index ${prototypeIndex.size}`;
	});
</script>

<div class="fixed inset-0 z-40 overflow-hidden bg-black text-white">
	<video
		bind:this={video}
		class="absolute inset-0 size-full object-cover"
		autoplay
		muted
		playsinline
	></video>

	{#if phase === 'scanning' && frame}
		<svg
			class="pointer-events-none absolute inset-0 z-10 size-full"
			viewBox={`0 0 ${viewport.width} ${viewport.height}`}
			aria-hidden="true"
		>
			{#each frame.regions.filter((r) => r.misses === 0 && !r.nested) as region (region.id)}
				{@const box = toScreen(region.box)}
				{@const matched = region.artworkId !== null}
				{@const isCurrent = currentId !== null && region.artworkId === currentId}
				<rect
					x={box.x1}
					y={box.y1}
					width={Math.max(0, box.x2 - box.x1)}
					height={Math.max(0, box.y2 - box.y1)}
					rx="14"
					fill={isCurrent ? 'rgba(232, 118, 56, 0.08)' : 'none'}
					stroke={isCurrent
						? 'rgb(232, 118, 56)'
						: matched
							? 'rgba(255,255,255,0.75)'
							: 'rgba(255,255,255,0.35)'}
					stroke-width={isCurrent ? 3 : 1.5}
					stroke-dasharray={matched ? undefined : '6 8'}
					class="transition-all duration-200 ease-out"
				/>
			{/each}
		</svg>
	{/if}

	<div
		class="absolute inset-x-0 top-0 z-20 flex items-center justify-between p-4 pt-[max(1rem,env(safe-area-inset-top))]"
	>
		<a
			href={backHref}
			class="grid size-10 place-items-center rounded-full bg-black/40 backdrop-blur focus-visible:focus-ring"
			aria-label="Close scanner"
		>
			<X class="size-5" aria-hidden="true" />
		</a>
		<p class="text-xs font-semibold tracking-[0.2em] text-white/70 uppercase">{title}</p>
		<span class="w-10 text-right text-[10px] text-white/40 tabular-nums" aria-hidden="true"
			>{phase === 'scanning' ? `${fps} fps` : ''}</span
		>
	</div>

	{#if debug && phase === 'scanning'}
		<p
			class="absolute inset-x-0 top-16 z-30 px-4 text-center font-mono text-[11px] text-white/80"
			data-debug
		>
			{debugLine}
		</p>
	{/if}

	{#if phase === 'scanning' && !current && !unknown}
		<div
			class="pointer-events-none absolute inset-0 z-10 grid place-items-center"
			transition:fade={{ duration: 250 }}
		>
			<svg
				viewBox="0 0 100 64"
				class="w-[min(70vmin,480px)] animate-pulse opacity-90"
				aria-hidden="true"
			>
				<g
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					vector-effect="non-scaling-stroke"
				>
					<path d="M8 18 L8 8 L28 8" /><path d="M92 18 L92 8 L72 8" /><path
						d="M8 46 L8 56 L28 56"
					/><path d="M92 46 L92 56 L72 56" />
				</g>
			</svg>
			{#key promptIndex}
				<p
					class="absolute max-w-[26ch] px-6 text-center font-display text-2xl font-medium text-white/90 drop-shadow"
					in:fade={{ duration: 300 }}
					aria-live="polite"
				>
					{prompts[promptIndex]}
				</p>
			{/key}
		</div>
	{/if}

	{#if unknown && phase === 'scanning'}
		<div
			class="absolute inset-x-4 bottom-[max(1.25rem,env(safe-area-inset-bottom))] z-20 sm:inset-x-auto sm:right-6 sm:bottom-6 sm:w-96"
			transition:fly={{ y: 16, duration: 250 }}
		>
			<div
				class="flex items-center gap-3 rounded-3xl border border-white/15 bg-neutral-950/60 p-3 pl-5 shadow-lift backdrop-blur-xl"
			>
				<div class="min-w-0 flex-1">
					<p class="font-display text-lg font-medium">Not in your collection yet</p>
					<p class="text-xs text-white/60">Add it now with this photo as its first picture.</p>
				</div>
				<Button variant="inverse" size="sm" onclick={addUnknown} disabled={adding}>
					<Plus class="size-4" aria-hidden="true" />
					{adding ? 'Saving…' : 'Add piece'}
				</Button>
			</div>
		</div>
	{/if}

	{#if current && currentId}
		<div class="absolute inset-0 z-20" transition:fly={{ y: 24, duration: 300 }}>
			<ScannerOverlay artwork={current} artworkId={currentId} {reactions} />
		</div>
	{/if}

	{#if phase === 'loading'}
		<div
			class="absolute inset-0 z-30 grid place-items-center bg-black/70 backdrop-blur-sm"
			role="status"
		>
			<div class="flex flex-col items-center gap-3 px-6 text-center">
				<span class="size-8 animate-spin rounded-full border-2 border-white/30 border-t-white"
				></span>
				<p class="text-sm text-white/80">{status}…</p>
				<p class="text-xs text-white/40">{manifest.config.attribution}</p>
			</div>
		</div>
	{/if}

	{#if phase === 'permission' || phase === 'error' || phase === 'idle'}
		<div class="absolute inset-0 z-30 grid place-items-center bg-neutral-950/95 p-6">
			<div
				class="w-full max-w-sm rounded-3xl border border-white/15 bg-white/5 p-8 text-center shadow-lift backdrop-blur-xl"
			>
				<div class="mx-auto grid size-14 place-items-center rounded-2xl bg-white/10">
					<Camera class="size-6" aria-hidden="true" />
				</div>
				{#if phase === 'error'}
					<h2 class="mt-5 font-display text-2xl font-medium">Can’t scan right now</h2>
					<p class="mt-2 text-sm text-white/70">{errorMessage}</p>
					<div class="mt-6"><Button variant="inverse" onclick={start}>Try again</Button></div>
				{:else}
					<h2 class="mt-5 font-display text-2xl font-medium">Turn on your camera</h2>
					<p class="mt-2 text-sm text-white/70">
						{permission === 'denied'
							? 'Camera access is blocked. Allow it in your browser settings, then try again.'
							: 'Point your phone at a piece and its story appears. Frames never leave this device.'}
					</p>
					<div class="mt-6">
						<Button variant="inverse" size="lg" onclick={start}>Enable camera</Button>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>
