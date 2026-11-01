<script lang="ts">
	import { enhance } from '$app/forms';
	import { Button, Card, Field, Input } from '#lib/components/index.js';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();
	let submitting = $state(false);
</script>

<svelte:head><title>Sign in · Collective</title></svelte:head>

<h1 class="font-display text-3xl font-medium tracking-tight">Welcome back</h1>
<p class="mt-2 text-sm text-ink-soft">Sign in to your collection.</p>

<Card class="mt-8">
	<form
		method="POST"
		class="flex flex-col gap-5"
		use:enhance={() => {
			submitting = true;
			return async ({ update }) => {
				await update();
				submitting = false;
			};
		}}
	>
		<input type="hidden" name="next" value={data.next} />
		{#if form?.formError}
			<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
				{form.formError}
			</p>
		{/if}
		<Field label="Email" name="email" error={form?.errors.email}>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="email"
					type="email"
					autocomplete="email"
					required
					value={form?.values.email ?? ''}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
		<Field label="Password" name="password" error={form?.errors.password}>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="password"
					type="password"
					autocomplete="current-password"
					required
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
		<Button type="submit" size="lg" loading={submitting} class="mt-1">Sign in</Button>
	</form>
</Card>

<p class="mt-6 text-center text-sm text-ink-soft">
	New here? <a href="/signup" class="font-medium text-ink underline-offset-4 hover:underline"
		>Create an account</a
	>
</p>
