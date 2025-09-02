// Use global THREE and MindARThree set by index.html
const { THREE, MindARThree } = window;
import { buildOverlay } from './overlays.js';

export async function initTracker({ map, content, versions }) {
  const mindarThree = new MindARThree({
    container: document.body,
    imageTargetSrc: `${import.meta.env.BASE_URL}targets/${versions.mindFile}`,
    maxTrack: versions.maxTrack,
    filterMinCF: 0.01,     // Even more jitter reduction (default: 0.001)
    filterBeta: 5,         // More smoothing (default: 1000)
    warmupTolerance: 3,    // Frames to confirm detection (default: 5)
    missTolerance: 3       // Frames to confirm loss (default: 5)
  });

  const { renderer, scene, camera } = mindarThree;
  
  // Configure renderer for video background
  renderer.setClearColor(0x000000, 0); // Transparent background
  renderer.autoClear = true; // Allow proper clearing for overlay visibility

  // Add lighting
  const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
  scene.add(light);

  // Get HTML overlay element
  const artworkInfo = document.getElementById('artwork-info');
  const titleEl = document.getElementById('artwork-title');
  const artistEl = document.getElementById('artwork-artist');
  const materialsEl = document.getElementById('artwork-materials');
  const descriptionEl = document.getElementById('artwork-description');

  // Create anchors for each target
  for (const { index, artId } of map.targets) {
    const anchor = mindarThree.addAnchor(index);
    const group = await buildOverlay(content[artId]);
    group.visible = false;
    anchor.group.add(group);

    const artData = content[artId];

    anchor.onTargetFound = () => {
      // Show AR animation
      group.visible = true;
      group.traverse((child) => {
        child.visible = true;
      });
      
      // Populate and show HTML overlay
      titleEl.textContent = artData.title || '';
      artistEl.textContent = `${artData.artist || ''}${artData.year ? ', ' + artData.year : ''}`;
      materialsEl.textContent = artData.materials || '';
      descriptionEl.textContent = artData.description || '';
      artworkInfo.classList.add('visible');
    };
    
    anchor.onTargetLost = () => {
      // Hide AR animation
      group.visible = false;
      group.traverse((child) => {
        child.visible = false;
      });
      
      // Hide HTML overlay
      artworkInfo.classList.remove('visible');
    };
  }

  // Handle window resize
  window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  });

  await mindarThree.start();
  
  // Ensure video background is visible
  const videos = document.querySelectorAll('video');
  const canvases = document.querySelectorAll('canvas');
  
  if (videos.length > 0) {
    const video = videos[0];
    setTimeout(() => {
      video.style.setProperty('z-index', '0', 'important');
      video.style.setProperty('visibility', 'visible', 'important');
      video.style.setProperty('display', 'block', 'important');
    }, 100);
  }
  
  if (canvases.length > 0) {
    const canvas = canvases[0];
    canvas.style.setProperty('z-index', '1', 'important');
    canvas.style.setProperty('background', 'transparent', 'important');
  }
  
  // Start render loop after MindAR is initialized
  const clock = new THREE.Clock();
  
  const animate = () => {
    requestAnimationFrame(animate);
    
    const time = clock.getElapsedTime();
    
    // Update shader uniforms for all visible glow effects
    scene.traverse((object) => {
      if (object.userData?.uniforms && object.visible) {
        object.userData.uniforms.u_time.value = time;
      }
    });
    
    renderer.render(scene, camera);
  };
  animate();
}

function fade(group, show) {
  group.visible = show;
  
  // Ensure materials are properly hidden/shown
  group.traverse(child => {
    if (child.material) {
      if (show) {
        child.material.opacity = child.material.userData?.originalOpacity || 0.8;
        child.material.transparent = true;
        child.visible = true;
      } else {
        // Store original opacity and hide completely
        if (!child.material.userData) child.material.userData = {};
        child.material.userData.originalOpacity = child.material.opacity;
        child.material.opacity = 0;
        child.visible = false;
      }
    }
  });
}