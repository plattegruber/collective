<script lang="ts">
	import { page } from '$app/state';
	import { Frame, LayoutGrid, Plus, ScanLine, Settings } from 'lucide-svelte';
	import Logo from '#lib/components/Logo.svelte';
	import type { LayoutProps } from './$types';

	let { data, children }: LayoutProps = $props();

	const nav = [
		{ href: '/app', label: 'Overview', icon: LayoutGrid, exact: true },
		{ href: '/app/artworks', label: 'Artworks', icon: Frame, exact: false },
		{ href: '/app/scan', label: 'Scan', icon: ScanLine, exact: false },
		{ href: '/app/settings', label: 'Settings', icon: Settings, exact: false }
	];

	const isActive = (href: string, exact: boolean) =>
		exact ? page.url.pathname === href : page.url.pathname.startsWith(href);
</script>

<div class="flex min-h-full flex-col bg-paper lg:flex-row">
	<aside
		class="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-paper/90 px-4 py-3 backdrop-blur lg:h-screen lg:w-64 lg:flex-col lg:items-stretch lg:justify-start lg:border-r lg:border-b-0 lg:px-5 lg:py-6"
	>
		<a href="/app" class="rounded-lg focus-visible:focus-ring"><Logo /></a>

		<nav class="hidden lg:mt-10 lg:flex lg:flex-1 lg:flex-col lg:gap-1" aria-label="Primary">
			{#each nav as item (item.href)}
				{@const active = isActive(item.href, item.exact)}
				<a
					href={item.href}
					aria-current={active ? 'page' : undefined}
					class={[
						'flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition focus-visible:focus-ring',
						active
							? 'bg-ink text-paper shadow-card'
							: 'text-ink-soft hover:bg-paper-deep hover:text-ink'
					].join(' ')}
				>
					<item.icon class="size-4" aria-hidden="true" />
					{item.label}
				</a>
			{/each}
			<a
				href="/app/artworks/new"
				class="mt-4 flex items-center gap-3 rounded-xl border border-dashed border-line px-3 py-2 text-sm font-medium text-ink-soft transition hover:border-ink/40 hover:text-ink focus-visible:focus-ring"
			>
				<Plus class="size-4" aria-hidden="true" />
				Add artwork
			</a>
		</nav>

		<div class="flex items-center gap-3 lg:border-t lg:border-line lg:pt-4">
			<div class="hidden min-w-0 flex-1 lg:block">
				<p class="truncate text-sm font-medium text-ink">{data.user.name}</p>
				<p class="truncate text-xs text-ink-mute">{data.user.email}</p>
			</div>
			<form method="POST" action="/logout">
				<button
					type="submit"
					class="rounded-lg px-2 py-1 text-xs font-medium text-ink-soft hover:bg-paper-deep hover:text-ink focus-visible:focus-ring"
				>
					Sign out
				</button>
			</form>
		</div>
	</aside>

	<main class="flex-1 px-4 pt-6 pb-28 sm:px-8 lg:px-12 lg:py-10">
		<div class="mx-auto w-full max-w-6xl">{@render children()}</div>
	</main>

	<nav
		class="fixed inset-x-0 bottom-0 z-30 grid grid-cols-4 border-t border-line bg-paper/95 pb-[env(safe-area-inset-bottom)] backdrop-blur lg:hidden"
		aria-label="Primary"
	>
		{#each nav as item (item.href)}
			{@const active = isActive(item.href, item.exact)}
			<a
				href={item.href}
				aria-current={active ? 'page' : undefined}
				class={[
					'flex flex-col items-center gap-1 py-2.5 text-[11px] font-medium',
					active ? 'text-ink' : 'text-ink-mute'
				].join(' ')}
			>
				<item.icon class="size-5" aria-hidden="true" />
				{item.label}
			</a>
		{/each}
	</nav>
</div>
