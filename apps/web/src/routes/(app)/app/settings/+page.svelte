<script lang="ts">
	import { enhance } from '$app/forms';
	import { invalidateAll } from '$app/navigation';
	import { page } from '$app/state';
	import { Check, Copy, KeyRound, ScanLine } from 'lucide-svelte';
	import { Button, Card, Field, Input, Textarea } from '#lib/components/index.js';
	import { formatLoadStage } from '#lib/recognition/loader.js';
	import { formatRelative } from '#lib/utils/format.js';
	import { rebuildIndex, type RebuildProgress } from '#lib/recognition/teach.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import type { PageProps } from './$types';

	let { data, form }: PageProps = $props();

	const errors: Record<string, string> = $derived((form && 'errors' in form && form.errors) || {});
	const values: Record<string, string> = $derived((form && 'values' in form && form.values) || {});
	const formError = $derived(form && 'formError' in form ? form.formError : undefined);
	const newToken = $derived(form && 'newToken' in form ? form.newToken : null);
	const origin = $derived(page.url.origin);

	let copied = $state(false);
	let confirmingDelete = $state(false);

	$effect(() => {
		if (!form) return;
		if ('profileSaved' in form) toasts.push('Profile saved.', 'success');
		if ('gallerySaved' in form) toasts.push('Gallery settings saved.', 'success');
		if ('passwordChanged' in form)
			toasts.push('Password changed. Other devices were signed out.', 'success');
		if ('tokenRevoked' in form) toasts.push('Token revoked.');
		if ('message' in form && form.message) toasts.push(form.message, 'error');
	});

	async function copyToken() {
		if (!newToken) return;
		await navigator.clipboard.writeText(newToken);
		copied = true;
		setTimeout(() => (copied = false), 2000);
	}

	let rebuild = $state<RebuildProgress | null>(null);
	let rebuilding = $state(false);

	async function runRebuild() {
		rebuilding = true;
		try {
			const result = await rebuildIndex((progress) => (rebuild = progress));
			toasts.push(
				result.failed > 0
					? `Indexed ${result.done} photos; ${result.failed} failed.`
					: `Indexed ${result.done} photo${result.done === 1 ? '' : 's'}.`,
				result.failed > 0 ? 'error' : 'success'
			);
			await invalidateAll();
		} catch (cause) {
			toasts.push(cause instanceof Error ? cause.message : 'Could not rebuild the index.', 'error');
		} finally {
			rebuilding = false;
			rebuild = null;
		}
	}
</script>

<svelte:head><title>Settings · Collective</title></svelte:head>

<h1 class="font-display text-3xl font-medium tracking-tight sm:text-4xl">Settings</h1>

<div class="mt-8 flex max-w-3xl flex-col gap-8">
	<Card id="profile" class="flex flex-col gap-5">
		<h2 class="font-display text-xl font-medium">Profile</h2>
		<form method="POST" action="?/profile" class="flex flex-col gap-4" use:enhance>
			<Field label="Name" name="name" error={errors.name}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="name"
						value={values.name ?? data.user.name}
						required
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<p class="text-sm text-ink-mute">Signed in as {data.user.email}</p>
			<div><Button type="submit" variant="secondary">Save profile</Button></div>
		</form>
	</Card>

	<Card id="gallery" class="flex flex-col gap-5">
		<div>
			<h2 class="font-display text-xl font-medium">Public gallery</h2>
			<p class="mt-1 text-sm text-ink-soft">
				A shareable page with the pieces you mark public, plus the phone scanner for guests.
			</p>
		</div>
		<form method="POST" action="?/gallery" class="flex flex-col gap-4" use:enhance>
			{#if formError && 'handle' in errors}
				<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
					{formError}
				</p>
			{/if}
			<Field
				label="Handle"
				name="handle"
				hint={`Your gallery lives at ${origin}/g/your-handle`}
				error={errors.handle}
			>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="handle"
						value={values.handle ?? data.user.handle ?? ''}
						placeholder="your-name"
						autocapitalize="off"
						spellcheck={false}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Gallery title" name="galleryTitle" optional error={errors.galleryTitle}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="galleryTitle"
						value={values.galleryTitle ?? data.user.galleryTitle ?? ''}
						placeholder={`${data.user.name}'s collection`}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<Field label="Introduction" name="galleryIntro" optional error={errors.galleryIntro}>
				{#snippet children({ id, describedBy, invalid })}
					<Textarea
						{id}
						name="galleryIntro"
						value={values.galleryIntro ?? data.user.galleryIntro ?? ''}
						rows={3}
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<label
				class="flex cursor-pointer items-start gap-3 rounded-xl border border-line p-3 hover:bg-paper-deep/50"
			>
				<input
					type="checkbox"
					name="galleryPublic"
					class="mt-0.5 size-4 rounded border-line text-ink focus:ring-accent/40"
					checked={values.galleryPublic ? values.galleryPublic === 'on' : data.user.galleryPublic}
				/>
				<span>
					<span class="block text-sm font-medium">Make my gallery public</span>
					<span class="block text-xs text-ink-mute"
						>{data.stats.publicCount} of {data.stats.total} pieces are marked public.</span
					>
				</span>
			</label>
			<div class="flex items-center gap-3">
				<Button type="submit" variant="secondary">Save gallery</Button>
				{#if data.user.galleryPublic && data.user.handle}
					<a
						href={`/g/${data.user.handle}`}
						class="text-sm font-medium text-ink underline-offset-4 hover:underline">Open gallery</a
					>
				{/if}
			</div>
		</form>
	</Card>

	<Card id="recognizer" class="flex flex-col gap-5">
		<div class="flex items-start gap-3">
			<ScanLine class="mt-1 size-5 shrink-0 text-accent-deep" aria-hidden="true" />
			<div>
				<h2 class="font-display text-xl font-medium">Recognizer</h2>
				<p class="mt-1 text-sm text-ink-soft">
					Runs entirely on your device. Photos are indexed in your browser the moment you upload
					them, and the scanner keeps learning the views it sees on your walls. Nothing to train.
				</p>
			</div>
		</div>
		{#if !data.recognizer}
			<p class="rounded-xl bg-paper-deep/60 px-4 py-3 text-sm text-ink-soft">
				No recognizer release has been published for this installation yet.
			</p>
		{:else}
			{@const status = data.recognizer.status}
			<dl class="grid gap-3 text-sm sm:grid-cols-3">
				<div class="rounded-xl bg-paper-deep/60 px-4 py-3">
					<dt class="text-xs text-ink-mute">Release</dt>
					<dd class="mt-0.5 font-medium">{data.recognizer.manifest.version}</dd>
				</div>
				<div class="rounded-xl bg-paper-deep/60 px-4 py-3">
					<dt class="text-xs text-ink-mute">Photos indexed</dt>
					<dd class="mt-0.5 font-medium">{status.indexed} of {status.images}</dd>
				</div>
				<div class="rounded-xl bg-paper-deep/60 px-4 py-3">
					<dt class="text-xs text-ink-mute">Views learned while scanning</dt>
					<dd class="mt-0.5 font-medium">{status.memories}</dd>
				</div>
			</dl>
			{#if status.pending.length > 0}
				<div class="flex flex-wrap items-center gap-3">
					<Button type="button" variant="secondary" onclick={runRebuild} disabled={rebuilding}>
						{rebuilding
							? 'Indexing…'
							: `Index ${status.pending.length} photo${status.pending.length === 1 ? '' : 's'}`}
					</Button>
					{#if rebuild}
						<p class="text-sm text-ink-soft" aria-live="polite">
							{#if rebuild.stage && rebuild.done === 0}{formatLoadStage(
									rebuild.stage
								)}…{:else}{rebuild.done}
								of {rebuild.total} done{/if}
						</p>
					{/if}
				</div>
			{/if}
			<p class="text-xs text-ink-mute">{data.recognizer.manifest.config.attribution}</p>
		{/if}
	</Card>

	<Card id="tokens" class="flex flex-col gap-5">
		<div class="flex items-start gap-3">
			<KeyRound class="mt-1 size-5 shrink-0 text-accent-deep" aria-hidden="true" />
			<div>
				<h2 class="font-display text-xl font-medium">API tokens</h2>
				<p class="mt-1 text-sm text-ink-soft">
					Let tools such as <code class="rounded bg-paper-deep px-1 text-xs">collective-ml</code> read
					your collection through the API. Treat tokens like passwords.
				</p>
			</div>
		</div>
		{#if newToken}
			<div class="rounded-xl border border-success/30 bg-success-soft p-4">
				<p class="text-sm font-medium text-success">
					Copy this token now. It will not be shown again.
				</p>
				<div class="mt-2 flex items-center gap-2">
					<code class="min-w-0 flex-1 truncate rounded-lg bg-white px-3 py-2 font-mono text-xs"
						>{newToken}</code
					>
					<Button type="button" size="sm" variant="secondary" onclick={copyToken}>
						{#if copied}<Check class="size-4" aria-hidden="true" /> Copied{:else}<Copy
								class="size-4"
								aria-hidden="true"
							/> Copy{/if}
					</Button>
				</div>
				<pre
					class="mt-3 overflow-x-auto rounded-lg bg-ink p-3 text-xs text-paper">export COLLECTIVE_URL={origin}
export COLLECTIVE_TOKEN={newToken}
collective-ml train-and-publish</pre>
			</div>
		{/if}
		{#if data.tokens.length > 0}
			<ul class="divide-y divide-line">
				{#each data.tokens as token (token.id)}
					<li class="flex items-center gap-3 py-3">
						<div class="min-w-0 flex-1">
							<p class="text-sm font-medium">
								{token.name} <span class="font-mono text-xs text-ink-mute">{token.prefix}…</span>
							</p>
							<p class="text-xs text-ink-mute">
								Created {formatRelative(token.createdAt)}{#if token.lastUsedAt}
									· last used {formatRelative(token.lastUsedAt)}{/if}
							</p>
						</div>
						<form method="POST" action="?/revokeToken" use:enhance>
							<input type="hidden" name="id" value={token.id} /><Button
								type="submit"
								size="sm"
								variant="ghost">Revoke</Button
							>
						</form>
					</li>
				{/each}
			</ul>
		{/if}
		<form
			method="POST"
			action="?/createToken"
			class="flex flex-col gap-3 sm:flex-row sm:items-end"
			use:enhance
		>
			<div class="flex-1">
				<Field label="Token name" name="name" optional error={errors.name}>
					{#snippet children({ id, describedBy, invalid })}
						<Input
							{id}
							name="name"
							placeholder="Training laptop"
							aria-describedby={describedBy}
							{invalid}
						/>
					{/snippet}
				</Field>
			</div>
			<Button type="submit" variant="secondary">Create token</Button>
		</form>
	</Card>

	<Card id="export" class="flex flex-col gap-3">
		<h2 class="font-display text-xl font-medium">Export</h2>
		<p class="text-sm text-ink-soft">
			Download your whole inventory as a spreadsheet, including prices and private notes. Your data
			is yours.
		</p>
		<div><Button href="/app/export.csv" variant="secondary" download>Download CSV</Button></div>
	</Card>

	<Card id="password" class="flex flex-col gap-5">
		<h2 class="font-display text-xl font-medium">Password</h2>
		<form method="POST" action="?/password" class="flex flex-col gap-4" use:enhance>
			{#if formError && 'current' in errors}
				<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
					{formError}
				</p>
			{/if}
			<Field label="Current password" name="current" error={errors.current}>
				{#snippet children({ id, describedBy, invalid })}
					<Input
						{id}
						name="current"
						type="password"
						autocomplete="current-password"
						required
						aria-describedby={describedBy}
						{invalid}
					/>
				{/snippet}
			</Field>
			<div class="grid gap-4 sm:grid-cols-2">
				<Field label="New password" name="password" error={errors.password}>
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
				<Field label="Confirm" name="confirm" error={errors.confirm}>
					{#snippet children({ id, describedBy, invalid })}
						<Input
							{id}
							name="confirm"
							type="password"
							autocomplete="new-password"
							required
							aria-describedby={describedBy}
							{invalid}
						/>
					{/snippet}
				</Field>
			</div>
			<div><Button type="submit" variant="secondary">Change password</Button></div>
		</form>
	</Card>
	<Card id="danger" class="flex flex-col gap-4 border-danger/30">
		<div>
			<h2 class="font-display text-xl font-medium">Delete account</h2>
			<p class="mt-1 text-sm text-ink-soft">
				Removes your account, every artwork, photo and model, permanently. Export a CSV first if you
				want a copy.
			</p>
		</div>
		{#if confirmingDelete}
			<form method="POST" action="?/deleteAccount" class="flex flex-col gap-4" use:enhance>
				{#if formError && 'password' in errors && !('current' in errors)}
					<p class="rounded-xl bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
						{formError}
					</p>
				{/if}
				<div class="grid gap-4 sm:grid-cols-2">
					<Field label="Your password" name="password" error={errors.password}>
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
					<Field label="Type DELETE to confirm" name="confirm" error={errors.confirm}>
						{#snippet children({ id, describedBy, invalid })}
							<Input
								{id}
								name="confirm"
								autocomplete="off"
								autocapitalize="characters"
								required
								aria-describedby={describedBy}
								{invalid}
							/>
						{/snippet}
					</Field>
				</div>
				<div class="flex gap-2">
					<Button type="submit" variant="danger">Delete my account</Button>
					<Button type="button" variant="ghost" onclick={() => (confirmingDelete = false)}
						>Keep it</Button
					>
				</div>
			</form>
		{:else}
			<div>
				<Button
					type="button"
					variant="ghost"
					class="text-danger hover:bg-danger-soft hover:text-danger"
					onclick={() => (confirmingDelete = true)}
				>
					Delete account…
				</Button>
			</div>
		{/if}
	</Card>
</div>
