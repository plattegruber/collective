<script>
  export let visible = false;
  export let message = '';
</script>

{#if visible}
  <div class="loading">{message}</div>
{/if}

<style>
  .loading {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    color: #fff;
    font-family: system-ui, sans-serif;
    z-index: 50;
    background: rgba(0, 0, 0, 0.6);
  }
</style>
