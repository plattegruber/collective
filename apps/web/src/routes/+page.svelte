<script lang="ts">
	import { page } from '$app/state';
	import { ArrowRight } from 'lucide-svelte';
	import Button from '#lib/components/Button.svelte';
	import {
		ArtTile,
		Faq,
		FeatureList,
		HowItWorks,
		MockArtworkDetail,
		MockCollectionGrid,
		MockDashboard,
		MockScanner,
		Screenshots,
		Section,
		SiteFooter,
		SiteHeader
	} from '#lib/marketing/index.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const signedIn = $derived(Boolean(data.user));
	const title = 'Collective — the inventory for the art you live with';
	const description =
		'Catalog the art you own with photos, provenance, and value. Share a curated gallery, and let guests point a phone at any piece to learn its story.';
	const ogImage = $derived(`${page.url.origin}/marketing/og.jpg`);
</script>

<svelte:head>
	<title>{title}</title>
	<meta name="description" content={description} />
	<meta property="og:type" content="website" />
	<meta property="og:title" content={title} />
	<meta property="og:description" content={description} />
	<meta property="og:url" content={page.url.origin} />
	<meta property="og:image" content={ogImage} />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={description} />
	<meta name="twitter:image" content={ogImage} />
</svelte:head>

<div class="flex min-h-full flex-col bg-paper">
	<SiteHeader {signedIn} />

	<main id="main" class="flex-1">
		<section class="relative overflow-hidden" aria-labelledby="hero-title">
			<div
				class="pointer-events-none absolute inset-x-0 top-0 h-[32rem] bg-[radial-gradient(60%_60%_at_70%_0%,var(--color-accent-soft),transparent_70%)]"
				aria-hidden="true"
			></div>
			<div
				class="relative mx-auto grid max-w-6xl items-center gap-12 px-5 pt-16 pb-12 sm:px-8 sm:pt-24 lg:grid-cols-[1.1fr_1fr] lg:pb-24"
			>
				<div class="max-w-xl">
					<p
						class="inline-flex items-center gap-2 rounded-full border border-line bg-card px-3 py-1 text-xs font-medium text-ink-soft"
					>
						<span class="size-1.5 rounded-full bg-success" aria-hidden="true"></span>
						Free during early access
					</p>
					<h1
						id="hero-title"
						class="mt-6 font-display text-5xl leading-[1.02] font-medium tracking-tight text-balance sm:text-6xl lg:text-7xl"
					>
						The inventory for the art you live with.
					</h1>
					<p class="mt-6 max-w-lg text-lg leading-8 text-ink-soft">
						Collective keeps every piece you own in one calm place: photos, provenance, what you
						paid, what it's worth, and where it hangs. Then it does something no spreadsheet can: it
						recognizes your art on the wall.
					</p>
					<div class="mt-8 flex flex-wrap items-center gap-3">
						{#if signedIn}
							<Button href="/app" size="lg"
								>Open your collection <ArrowRight class="size-4" aria-hidden="true" /></Button
							>
						{:else}
							<Button href="/signup" size="lg"
								>Start your inventory <ArrowRight class="size-4" aria-hidden="true" /></Button
							>
							<Button href="/login" variant="secondary" size="lg">Sign in</Button>
						{/if}
					</div>
					<p class="mt-4 text-xs text-ink-mute">
						No credit card. Export anytime. Camera frames never leave your phone.
					</p>
				</div>

				<div class="relative">
					<div class="grid grid-cols-3 gap-3 sm:gap-4" aria-hidden="true">
						<ArtTile variant={0} framed class="aspect-[4/5] translate-y-6 rounded-sm" />
						<ArtTile variant={1} framed class="aspect-[4/5] -translate-y-2 rounded-sm" />
						<ArtTile variant={2} framed class="aspect-[4/5] translate-y-10 rounded-sm" />
						<ArtTile variant={3} framed class="aspect-[4/5] -translate-y-4 rounded-sm" />
						<ArtTile variant={4} framed class="aspect-[4/5] translate-y-2 rounded-sm" />
						<ArtTile variant={5} framed class="aspect-[4/5] translate-y-8 rounded-sm" />
					</div>
					<div
						class="absolute right-0 -bottom-6 left-0 mx-auto w-[88%] rounded-2xl border border-white/20 bg-[oklch(16%_0.01_280)]/85 p-4 text-white shadow-lift backdrop-blur-md sm:w-[70%]"
						aria-hidden="true"
					>
						<p class="text-[10px] font-semibold tracking-[0.18em] text-white/50 uppercase">
							Recognized
						</p>
						<p class="mt-1 font-display text-lg leading-tight font-semibold">Study in Blue</p>
						<p class="text-xs text-white/70">T. Okafor, 2021 · Acrylic on canvas</p>
					</div>
				</div>
			</div>
		</section>

		<Section
			id="inventory"
			eyebrow="Inventory"
			title="Every piece, every detail, found in seconds."
			description="Add a work in under a minute: only the title is required. Fill in the rest as you learn it. Search by artist, medium, or the room it's in; filter by status or tag."
		>
			<FeatureList
				items={[
					'Artist, year, medium, dimensions in cm or inches',
					'Acquisition date and source, purchase price, estimated value',
					'Location, provenance, private notes, tags',
					'Owned, on loan, sold, gifted, or wishlist'
				]}
			/>
			{#snippet aside()}<MockCollectionGrid />{/snippet}
		</Section>

		<Section
			id="details"
			eyebrow="Photos & provenance"
			title="A record you'd be proud to hand to an appraiser."
			description="Drop in phone photos; they're resized on your device and stored privately. Pick a cover, keep the rest. Provenance and notes stay with the piece forever."
			reverse
		>
			<FeatureList
				items={[
					'Up to twelve photos per piece with a chosen cover',
					'Images resized before upload, originals stay yours',
					'Nothing private ever appears on public pages'
				]}
			/>
			{#snippet aside()}<MockArtworkDetail />{/snippet}
		</Section>

		<Section
			id="overview"
			eyebrow="Overview"
			title="Know what you have, at a glance."
			description="The overview totals your collection by pieces, artists, estimated value, and how much is photographed. Multiple currencies are kept apart, never silently converted."
		>
			<FeatureList
				items={[
					'Estimated value with purchase price as fallback',
					'Pieces still missing a photo, so the model can learn them',
					'Recently added, one tap from the full grid'
				]}
			/>
			{#snippet aside()}<MockDashboard />{/snippet}
		</Section>

		<Section
			id="scanner"
			eyebrow="The scanner"
			title="Point a phone at the wall. Know the piece."
			description="A detector trained on your own photos runs inside the browser. Guests open your gallery link, allow the camera, and the story of whatever they're looking at appears, with a place to react."
			reverse
		>
			<FeatureList
				items={[
					'No app to install; works in Safari and Chrome',
					'Frames are processed on the phone and never uploaded',
					'Six emoji reactions, rate-limited and anonymous',
					'Share a curated public gallery at your own handle'
				]}
			/>
			{#snippet aside()}<MockScanner />{/snippet}
		</Section>

		<HowItWorks />
		<Screenshots />
		<Faq />

		<section class="py-16 sm:py-24" aria-labelledby="cta-title">
			<div class="mx-auto max-w-6xl px-5 sm:px-8">
				<div
					class="relative overflow-hidden rounded-card bg-ink px-6 py-14 text-paper shadow-lift sm:px-12 sm:py-20"
				>
					<div
						class="pointer-events-none absolute -top-24 -right-16 size-72 rounded-full bg-accent/40 blur-3xl"
						aria-hidden="true"
					></div>
					<div class="relative max-w-xl">
						<h2
							id="cta-title"
							class="font-display text-3xl font-medium tracking-tight text-balance sm:text-5xl"
						>
							Start with the piece you can see from your desk.
						</h2>
						<p class="mt-4 text-base leading-7 text-paper/75">
							One title, one photo. The rest can wait. Your future self, and whoever inherits the
							collection, will thank you.
						</p>
						<div class="mt-8 flex flex-wrap gap-3">
							{#if signedIn}
								<Button href="/app/artworks/new" variant="inverse" size="lg">Add an artwork</Button>
							{:else}
								<Button href="/signup" variant="inverse" size="lg">Create your account</Button>
								<Button
									href="/login"
									variant="ghost"
									size="lg"
									class="text-paper hover:bg-white/10 hover:text-paper">Sign in</Button
								>
							{/if}
						</div>
					</div>
				</div>
			</div>
		</section>
	</main>

	<SiteFooter />
</div>
