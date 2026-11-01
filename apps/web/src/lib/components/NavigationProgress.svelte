<script lang="ts">
	import { navigating } from '$app/state';

	// Only show the bar for navigations that take long enough to notice.
	let visible = $state(false);
	let timer: ReturnType<typeof setTimeout> | undefined;

	$effect(() => {
		if (navigating.to) {
			timer = setTimeout(() => (visible = true), 150);
		} else {
			clearTimeout(timer);
			visible = false;
		}
		return () => clearTimeout(timer);
	});
</script>

{#if visible}
	<div
		class="pointer-events-none fixed inset-x-0 top-0 z-[90] h-0.5 overflow-hidden bg-accent/20"
		role="progressbar"
		aria-label="Loading"
	>
		<div class="h-full w-1/3 animate-[slide_1s_ease-in-out_infinite] bg-accent"></div>
	</div>
{/if}

<style>
	@keyframes slide {
		from {
			transform: translateX(-100%);
		}
		to {
			transform: translateX(300%);
		}
	}
</style>
