<script lang="ts">
	import Badge from '#lib/components/Badge.svelte';
	import ArtTile from './ArtTile.svelte';

	const facts: [string, string][] = [
		['Medium', 'Oil on linen'],
		['Dimensions', '61 × 45.7 cm'],
		['Location', 'Living room, north wall'],
		['Acquired', 'Mar 2, 2021 · Studio visit'],
		['Purchase price', '$2,400.00'],
		['Estimated value', '$3,100.00']
	];
</script>

<div
	class="rounded-card border border-line bg-paper p-4 shadow-lift sm:p-5"
	role="img"
	aria-label="Mockup of an artwork detail page with photos, facts, and provenance"
>
	<div class="grid gap-4 sm:grid-cols-[1.1fr_1fr]">
		<div class="flex flex-col gap-2">
			<ArtTile variant={2} framed class="aspect-[4/5] rounded-sm" />
			<div class="flex gap-2">
				<ArtTile variant={2} class="size-12 rounded-lg border-2 border-ink" />
				<ArtTile variant={5} class="size-12 rounded-lg border border-line" />
				<ArtTile variant={0} class="size-12 rounded-lg border border-line" />
				<div
					class="grid size-12 place-items-center rounded-lg border border-dashed border-line text-[10px] text-ink-mute"
				>
					+ Add
				</div>
			</div>
		</div>
		<div class="flex min-w-0 flex-col gap-3">
			<div>
				<p class="text-[10px] font-semibold tracking-[0.18em] text-ink-mute uppercase">Artworks</p>
				<p class="font-display text-xl font-medium">Untitled (Kitchen)</p>
				<p class="text-xs text-ink-soft">Byron Anway, 2022</p>
				<div class="mt-2 flex flex-wrap gap-1.5">
					<Badge tone="success">Owned</Badge>
					<Badge tone="accent">Public</Badge>
					<span class="rounded-full border border-line px-2 py-0.5 text-[10px] text-ink-soft"
						>painting</span
					>
				</div>
			</div>
			<dl class="divide-y divide-line rounded-xl border border-line bg-card px-3">
				{#each facts as [label, value] (label)}
					<div class="grid grid-cols-[5.5rem_1fr] gap-2 py-1.5 text-[11px]">
						<dt class="text-ink-mute">{label}</dt>
						<dd class="truncate text-ink">{value}</dd>
					</div>
				{/each}
			</dl>
			<div class="rounded-xl border border-line bg-card px-3 py-2">
				<p class="text-[10px] font-semibold tracking-[0.16em] text-ink-mute uppercase">
					Provenance
				</p>
				<p class="mt-1 text-[11px] leading-5 text-ink-soft">
					Acquired directly from the artist after the “Rooms” show. Exhibited at Fold Gallery, 2022.
				</p>
			</div>
		</div>
	</div>
</div>
