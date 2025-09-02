// Use global THREE set by index.html
const { THREE } = window;

export async function buildOverlay(art) {
  const group = new THREE.Group();

  // Create simple animated circle that will rotate around artwork perimeter
  const circleGeo = new THREE.CircleGeometry(0.02, 16);
  const circleMat = new THREE.MeshBasicMaterial({ 
    color: 0x00ff88, 
    transparent: true,
    opacity: 0.9
  });
  const circle = new THREE.Mesh(circleGeo, circleMat);
  
  // Position circle at the edge of artwork (will be animated)
  const radius = 0.6; // Distance from center
  circle.position.set(radius, 0, 0.01);
  
  group.add(circle);
  group.userData = { circle, radius, angle: 0 };

  return group;
}

