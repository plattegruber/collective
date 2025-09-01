import { initTracker } from './tracker.js';

export const VERSIONS = {
  targetsMap: 'v1',
  artContent: 'v1',
  mindFile:   'gallery-v1.mind',
  maxTrack:   2
};

(async () => {
  try {
    const [map, content] = await Promise.all([
      fetch(`${import.meta.env.BASE_URL}targets/targets-map.${VERSIONS.targetsMap}.json`).then(r => r.json()),
      fetch(`${import.meta.env.BASE_URL}data/art-content.${VERSIONS.artContent}.json`).then(r => r.json())
    ]);

    await initTracker({ map, content, versions: VERSIONS });
    document.getElementById('loading').style.display = 'none';
  } catch (error) {
    console.error('Failed to initialize AR tracker:', error);
    document.getElementById('loading').textContent = 'Failed to load. Please refresh.';
  }
})();

