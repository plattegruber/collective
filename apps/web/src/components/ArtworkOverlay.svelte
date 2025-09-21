<script>
  export let visible = false;
  export let artwork = {
    title: '',
    byline: '',
    materials: '',
    description: '',
  };
</script>

<div class={`artwork-overlay ${visible ? 'visible' : ''}`}>
  <h2>{artwork.title}</h2>
  <p>{artwork.byline}</p>
  <p>{artwork.materials}</p>
  <p>{artwork.description}</p>
</div>

<style>
  .artwork-overlay {
    position: fixed;
    top: 20px;
    bottom: 20px;
    left: 20px;
    right: 20px;
    color: #f5f7ff;
    padding: clamp(20px, 4vw, 28px);
    border-radius: 18px;
    z-index: 10;
    pointer-events: none;
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.35s ease, transform 0.35s ease;
    background: rgba(15, 17, 26, 0.48);
    box-shadow: 0 28px 60px rgba(0, 0, 0, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(16px) saturate(125%);
    -webkit-backdrop-filter: blur(16px) saturate(125%);
    overflow: hidden;
  }

  @supports not (backdrop-filter: blur(1px)) {
    .artwork-overlay {
      background: rgba(12, 14, 20, 0.72);
    }
  }

  .artwork-overlay.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .artwork-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 40%);
    opacity: 0.9;
    pointer-events: none;
  }

  .artwork-overlay h2 {
    position: relative;
    font-size: clamp(1.5rem, 4vw, 2.2rem);
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.02em;
  }

  .artwork-overlay p {
    position: relative;
    margin: 6px 0;
    line-height: 1.65;
    color: rgba(240, 244, 255, 0.85);
  }

  .artwork-overlay p:last-child {
    margin-top: 16px;
    font-size: 0.95rem;
    color: rgba(224, 230, 255, 0.75);
  }
</style>
