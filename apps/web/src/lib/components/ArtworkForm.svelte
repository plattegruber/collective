<script lang="ts">
	import { untrack } from 'svelte';
	import { enhance } from '$app/forms';
	import { ARTWORK_STATUSES, DIMENSION_UNITS } from '#lib/domain/artwork.js';
	import { STATUS_LABELS } from '#lib/schemas/artwork.js';
	import { COMMON_CURRENCIES } from '#lib/utils/money.js';
	import Button from './Button.svelte';
	import Card from './Card.svelte';
	import Field from './Field.svelte';
	import Input from './Input.svelte';
	import Select from './Select.svelte';
	import Textarea from './Textarea.svelte';

	type Props = {
		values: Record<string, string>;
		errors?: Record<string, string>;
		formError?: string;
		submitLabel: string;
		cancelHref: string;
		tagSuggestions?: string[];
	};

	let {
		values,
		errors = {},
		formError,
		submitLabel,
		cancelHref,
		tagSuggestions = []
	}: Props = $props();

	let submitting = $state(false);
	let unit = $state(untrack(() => values.dimensionUnit ?? 'cm'));
	let currency = $state(untrack(() => values.currency ?? 'USD'));
	const v = (key: string) => values[key] ?? '';
</script>

<form
	method="POST"
	class="flex flex-col gap-6"
	use:enhance={() => {
		submitting = true;
		return async ({ update }) => {
			await update({ reset: false });
			submitting = false;
		};
	}}
>
	{#if formError}
		<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">{formError}</p>
	{/if}

	<Card class="flex flex-col gap-5">
		<h2 class="font-display text-xl font-medium">The piece</h2>
		<Field label="Title" name="title" error={errors.title}>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="title"
					value={v('title')}
					required
					maxlength={200}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
		<div class="grid gap-5 sm:grid-cols-[1fr_8rem]">
			<Field label="Artist" name="artist" optional error={errors.artist}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="artist"
						value={v('artist')}
						autocomplete="off"
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Year" name="year" optional error={errors.year}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="year"
						inputmode="numeric"
						value={v('year')}
						placeholder="1998"
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
		</div>
		<div class="grid gap-5 sm:grid-cols-2">
			<Field
				label="Date, as you'd print it"
				name="dateText"
				optional
				hint="Overrides the year on labels, e.g. “c. 1920” or “1998–2001”."
				error={errors.dateText}
			>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="dateText"
						value={v('dateText')}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Medium" name="medium" optional error={errors.medium}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="medium"
						value={v('medium')}
						placeholder="Oil on canvas"
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
		</div>
		<fieldset class="flex flex-col gap-2">
			<legend class="text-sm font-medium text-ink"
				>Dimensions <span class="font-normal text-ink-mute">(H × W × D)</span></legend
			>
			<div class="grid grid-cols-[1fr_1fr_1fr_5.5rem] gap-2">
				<Input
					name="height"
					inputmode="decimal"
					value={v('height')}
					placeholder="H"
					aria-label="Height"
					invalid={Boolean(errors.height)}
				/>
				<Input
					name="width"
					inputmode="decimal"
					value={v('width')}
					placeholder="W"
					aria-label="Width"
					invalid={Boolean(errors.width)}
				/>
				<Input
					name="depth"
					inputmode="decimal"
					value={v('depth')}
					placeholder="D"
					aria-label="Depth"
					invalid={Boolean(errors.depth)}
				/>
				<Select name="dimensionUnit" bind:value={unit} aria-label="Unit">
					{#each DIMENSION_UNITS as option (option)}<option value={option}>{option}</option>{/each}
				</Select>
			</div>
			{#if errors.height || errors.width || errors.depth}
				<p class="text-sm text-danger" role="alert">
					{errors.height ?? errors.width ?? errors.depth}
				</p>
			{/if}
		</fieldset>
		<Field
			label="Description"
			name="description"
			optional
			hint="Shown on labels and in the scanner overlay."
			error={errors.description}
		>
			{#snippet children({ id, describedBy, invalid })}
				<Textarea
					{id}
					name="description"
					value={v('description')}
					rows={4}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
	</Card>

	<Card class="flex flex-col gap-5">
		<h2 class="font-display text-xl font-medium">Ownership</h2>
		<div class="grid gap-5 sm:grid-cols-2">
			<Field label="Status" name="status" error={errors.status}>
				{#snippet children({ id, describedBy, invalid })}
					<Select
						{id}
						name="status"
						value={v('status') || 'owned'}
						aria-describedby={describedBy}
						{invalid}
					>
						{#each ARTWORK_STATUSES as status (status)}<option value={status}
								>{STATUS_LABELS[status]}</option
							>{/each}
					</Select>
				{/snippet}
			</Field>
			<Field
				label="Location"
				name="location"
				optional
				hint="Room, wall, storage box…"
				error={errors.location}
			>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="location"
						value={v('location')}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Acquired on" name="acquiredOn" optional error={errors.acquiredOn}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="acquiredOn"
						type="date"
						value={v('acquiredOn')}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field
				label="Acquired from"
				name="acquiredFrom"
				optional
				hint="Gallery, auction house, the artist…"
				error={errors.acquiredFrom}
			>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="acquiredFrom"
						value={v('acquiredFrom')}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
		</div>
		<div class="grid gap-5 sm:grid-cols-[1fr_1fr_7rem]">
			<Field label="Purchase price" name="purchasePrice" optional error={errors.purchasePrice}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="purchasePrice"
						inputmode="decimal"
						value={v('purchasePrice')}
						placeholder="0.00"
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Estimated value" name="estimatedValue" optional error={errors.estimatedValue}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="estimatedValue"
						inputmode="decimal"
						value={v('estimatedValue')}
						placeholder="0.00"
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Currency" name="currency" error={errors.currency}>
				{#snippet children({ id, describedBy, invalid })}
					<Select
						{id}
						name="currency"
						bind:value={currency}
						aria-describedby={describedBy}
						{invalid}
					>
						{#each COMMON_CURRENCIES as code (code)}<option value={code}>{code}</option>{/each}
						{#if !COMMON_CURRENCIES.includes(currency as (typeof COMMON_CURRENCIES)[number])}
							<option value={currency}>{currency}</option>
						{/if}
					</Select>
				{/snippet}
			</Field>
		</div>
		<Field
			label="Provenance"
			name="provenance"
			optional
			hint="Previous owners, exhibitions, publications."
			error={errors.provenance}
		>
			{#snippet children({ id, describedBy, invalid })}
				<Textarea
					{id}
					name="provenance"
					value={v('provenance')}
					rows={3}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
		<Field
			label="Private notes"
			name="notes"
			optional
			hint="Never shown publicly."
			error={errors.notes}
		>
			{#snippet children({ id, describedBy, invalid })}
				<Textarea
					{id}
					name="notes"
					value={v('notes')}
					rows={3}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
	</Card>

	<Card class="flex flex-col gap-5">
		<h2 class="font-display text-xl font-medium">Organize & share</h2>
		<Field
			label="Tags"
			name="tags"
			optional
			hint="Comma-separated. e.g. print, landscape, living room"
			error={errors.tags}
		>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="tags"
					value={v('tags')}
					list="tag-suggestions"
					aria-describedby={describedBy}
					{invalid}
				/>
				<datalist id="tag-suggestions">
					{#each tagSuggestions as tag (tag)}<option value={tag}></option>{/each}
				</datalist>
			{/snippet}
		</Field>
		<label
			class="flex cursor-pointer items-start gap-3 rounded-xl border border-line p-3 hover:bg-paper-deep/50"
		>
			<input
				type="checkbox"
				name="isPublic"
				class="mt-0.5 size-4 rounded border-line text-ink focus:ring-accent/40"
				checked={v('isPublic') === 'on' || v('isPublic') === 'true'}
			/>
			<span>
				<span class="block text-sm font-medium">Show in my public gallery</span>
				<span class="block text-xs text-ink-mute"
					>Only takes effect once sharing is enabled in Settings. Prices and notes never appear
					publicly.</span
				>
			</span>
		</label>
	</Card>

	<div
		class="sticky bottom-20 z-10 flex justify-end gap-3 rounded-2xl border border-line bg-paper/90 p-3 backdrop-blur lg:bottom-4"
	>
		<Button href={cancelHref} variant="ghost">Cancel</Button>
		<Button type="submit" loading={submitting}>{submitLabel}</Button>
	</div>
</form>
