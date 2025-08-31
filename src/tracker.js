// Use global THREE and MindARThree set by index.html
const { THREE, MindARThree } = window;
import { buildOverlay } from './overlays.js';

export async function initTracker({ map, content, versions }) {
  const mindarThree = new MindARThree({
    container: document.body,
    imageTargetSrc: `/targets/${versions.mindFile}`,
    maxTrack: versions.maxTrack
  });

  const { renderer, scene, camera } = mindarThree;

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
  console.log('MindAR started successfully');

  // Use the official MindAR render loop - this handles video background automatically!
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
  });
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