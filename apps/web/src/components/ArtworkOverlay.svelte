<script>
  export let visible = false;
  export let artwork = {
    title: '',
    byline: '',
    materials: '',
    description: '',
  };
</script>

<div
  class={`pointer-events-none fixed inset-5 z-10 overflow-hidden rounded-2xl border border-white/15 bg-slate-900/60 px-6 py-6 text-slate-100 shadow-[0_28px_60px_rgba(0,0,0,0.32)] backdrop-blur-xl transition-all duration-300 md:px-10 md:py-8 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-5'}`}
>
  <div class="absolute inset-0 -z-10 bg-gradient-to-tr from-white/10 to-transparent"></div>
  <h2 class="text-2xl font-semibold tracking-wide md:text-3xl">{artwork.title}</h2>
  <p class="mt-2 text-sm text-slate-200/90 md:text-base">{artwork.byline}</p>
  <p class="mt-2 text-sm text-slate-200/80 md:text-base">{artwork.materials}</p>
  <p class="mt-4 text-sm leading-relaxed text-slate-200/75 md:text-base">{artwork.description}</p>
</div>
