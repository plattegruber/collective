<script lang="ts">
	import { enhance } from '$app/forms';
	import { Button, Card, Field, Input } from '#lib/components/index.js';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();
	let submitting = $state(false);
</script>

<svelte:head><title>Create your account · Collective</title></svelte:head>

<h1 class="font-display text-3xl font-medium tracking-tight">Start your inventory</h1>
<p class="mt-2 text-sm text-ink-soft">Free for personal collections. Takes a minute.</p>

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
		{#if form?.formError}
			<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
				{form.formError}
			</p>
		{/if}
		<Field label="Your name" name="name" error={form?.errors.name}>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="name"
					autocomplete="name"
					required
					value={form?.values.name ?? ''}
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
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
		<Field
			label="Password"
			name="password"
			hint="At least 10 characters."
			error={form?.errors.password}
		>
			{#snippet children({ id, describedBy, invalid })}
				<Input
					{id}
					name="password"
					type="password"
					autocomplete="new-password"
					minlength={10}
					required
					aria-describedby={describedBy}
					{invalid}
				/>
			{/snippet}
		</Field>
		{#if data.inviteRequired}
			<Field label="Invite code" name="invite" error={form?.errors.invite}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="invite"
						required
						value={form?.values.invite ?? ''}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
		{/if}
		<Button type="submit" size="lg" loading={submitting} class="mt-1">Create account</Button>
	</form>
</Card>

<p class="mt-6 text-center text-sm text-ink-soft">
	Already have an account? <a
		href="/login"
		class="font-medium text-ink underline-offset-4 hover:underline">Sign in</a
	>
</p>
