<script lang="ts">
	import { fade, fly } from 'svelte/transition';
	import { Camera, X } from 'lucide-svelte';
	import { DEFAULT_THRESHOLD, loadDetector, type Detector } from '#lib/detector/runtime.js';
	import { DetectionTracker } from '#lib/detector/tracker.js';
	import type { DetectorManifest } from '#lib/detector/types.js';
	import {
		CameraError,
		openRearCamera,
		queryCameraPermission,
		stopStream,
		type CameraPermission
	} from './camera.js';
	import ScannerOverlay, { type ScannerArtwork } from './ScannerOverlay.svelte';
	import Button from '#lib/components/Button.svelte';

	type Props = {
		manifest: DetectorManifest;
		artworks: Record<string, ScannerArtwork>;
		reactions?: boolean;
		backHref: string;
		title?: string;
	};

	let { manifest, artworks, reactions = true, backHref, title = 'Scanner' }: Props = $props();

	type Phase = 'idle' | 'permission' | 'loading' | 'scanning' | 'error';
	let phase = $state<Phase>('idle');
	let status = $state('');
	let errorMessage = $state('');
	let permission = $state<CameraPermission>('unknown');
	let video: HTMLVideoElement | undefined = $state();
	let currentId = $state<string | null>(null);
	let fps = $state(0);
	/** `?debug` shows the strongest raw detection and the tracker's smoothed confidence; handy when tuning a model on site. */
	const debug =
		typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('debug');
	let debugLine = $state('');

	let stream: MediaStream | null = null;
	let detector: Detector | null = null;
	let frameHandle = 0;
	let running = false;
	/** Bumped by stop(); a start() that outlives its generation abandons its work. */
	let generation = 0;
	const tracker = new DetectionTracker();

	const current = $derived(currentId ? (artworks[currentId] ?? null) : null);
	const prompts = [
		'Point me toward the art.',
		'Frame the whole piece.',
		'A little closer.',
		'Hold steady for a moment.'
	];
	let promptIndex = $state(0);
	let promptTimer: ReturnType<typeof setInterval> | undefined;

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

			if (!detector) {
				const loaded = await loadDetector(manifest, (stage) => (status = stage));
				if (cancelled()) {
					void loaded.dispose();
					return;
				}
				detector = loaded;
			}
			tracker.reset();
			phase = 'scanning';
			running = true;
			promptTimer = setInterval(() => (promptIndex = (promptIndex + 1) % prompts.length), 4000);
			void loop();
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

	async function loop() {
		let lastTick = performance.now();
		let frames = 0;
		while (running && detector && video) {
			try {
				const detections = await detector.detect(video, debug ? 0.05 : undefined);
				const update = tracker.update(detections.filter((d) => d.score >= DEFAULT_THRESHOLD));
				if (update.changed) currentId = update.current;
				if (debug) {
					const best = detections[0];
					debugLine = best
						? `${artworks[best.artworkId]?.title ?? best.artworkId} raw ${best.score.toFixed(2)} · smoothed ${tracker.confidenceOf(best.artworkId).toFixed(2)}`
						: 'no detections';
				}
			} catch (cause) {
				console.error('Detection failed', cause);
			}
			frames += 1;
			const now = performance.now();
			if (now - lastTick >= 1000) {
				fps = frames;
				frames = 0;
				lastTick = now;
			}
			await new Promise<void>((resolve) => (frameHandle = requestAnimationFrame(() => resolve())));
		}
	}

	function stop() {
		generation += 1;
		running = false;
		cancelAnimationFrame(frameHandle);
		clearInterval(promptTimer);
		stopStream(stream);
		stream = null;
		currentId = null;
		phase = 'idle';
	}

	// $effect only runs in the browser, so none of this touches the camera or ORT during SSR.
	$effect(() => {
		let cancelled = false;
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
			stop();
			void detector?.dispose();
		};
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

	{#if phase === 'scanning' && !current}
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
			<div class="flex flex-col items-center gap-3">
				<span class="size-8 animate-spin rounded-full border-2 border-white/30 border-t-white"
				></span>
				<p class="text-sm text-white/80">{status}…</p>
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
