// Use global THREE and MindARThree set by index.html
const { THREE, MindARThree } = window;
import { buildOverlay } from './overlays.js';

export async function initTracker({ map, content, versions }) {
  const mindarThree = new MindARThree({
    container: document.body,
    imageTargetSrc: `${import.meta.env.BASE_URL}targets/${versions.mindFile}`,
    maxTrack: versions.maxTrack
  });

  const { renderer, scene, camera } = mindarThree;
  
  // Configure renderer for video background
  renderer.setClearColor(0x000000, 0); // Transparent background
  renderer.autoClear = false;

  // Add lighting
  const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
  scene.add(light);

  // Create anchors for each target
  for (const { index, artId } of map.targets) {
    const anchor = mindarThree.addAnchor(index);
    const group = await buildOverlay(content[artId]);
    group.visible = false;
    anchor.group.add(group);

    anchor.onTargetFound = () => {
      console.log(`Target found: ${artId}`);
      fade(group, true);
    };
    
    anchor.onTargetLost = () => {
      console.log(`Target lost: ${artId}`);
      fade(group, false);
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
  
  const animate = () => {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  };
  animate();
}

function fade(group, show) {
  // Simple visibility toggle - could be enhanced with smooth fade animation
  group.visible = show;
  
  // Optional: Add fade animation
  if (show) {
    group.traverse(child => {
      if (child.material && child.material.opacity !== undefined) {
        child.material.opacity = 1;
      }
    });
  }
}