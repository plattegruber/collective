<script>
  export let visible = false;
  export let message = '';
</script>

{#if visible}
  <div class="fixed inset-0 z-50 grid place-items-center bg-black/70 px-6 text-center text-lg font-semibold text-white">
    {message}
  </div>
{/if}
