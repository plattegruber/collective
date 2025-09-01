// Use global THREE set by index.html
const { THREE } = window;

export async function buildOverlay(art) {
  const group = new THREE.Group();

  // Panel background
  const { w, h, z } = art.overlay.panel;
  const panelGeo = new THREE.PlaneGeometry(w, h);
  const panelMat = new THREE.MeshBasicMaterial({ 
    color: 0x111111, 
    opacity: 0.8, 
    transparent: true 
  });
  const panel = new THREE.Mesh(panelGeo, panelMat);
  panel.position.set(0, 0, z); // Center on artwork
  group.add(panel);

  // Text content
  const textTexture = await createTextTexture(art);
  const textGeo = new THREE.PlaneGeometry(w * 0.96, h * 0.96);
  const textMat = new THREE.MeshBasicMaterial({ 
    map: textTexture, 
    transparent: true 
  });
  const textMesh = new THREE.Mesh(textGeo, textMat);
  textMesh.position.copy(panel.position);
  textMesh.position.z += 0.001; // Slightly forward to avoid z-fighting
  group.add(textMesh);

  // Optional thumbnail image - skip for now since assets don't exist
  // if (art.assets?.image) {
  //   try {
  //     const texture = await loadTexture(art.assets.image);
  //     const ratio = texture.image.width / texture.image.height;
  //     const imageWidth = Math.min(w * 0.3, 0.6);
  //     const imageHeight = imageWidth / ratio;
      
  //     const imageGeo = new THREE.PlaneGeometry(imageWidth, imageHeight);
  //     const imageMat = new THREE.MeshBasicMaterial({ map: texture });
  //     const imageMesh = new THREE.Mesh(imageGeo, imageMat);
  //     imageMesh.position.set(-w * 0.33, -h * 0.65, z + 0.002);
  //     group.add(imageMesh);
  //   } catch (error) {
  //     console.warn(`Failed to load image for ${art.title}:`, error);
  //   }
  // }

  return group;
}

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      texture => resolve(texture),
      undefined,
      error => reject(error)
    );
  });
}

async function createTextTexture(art) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Clear canvas with transparent background
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Set text properties
  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  // Title
  ctx.font = 'bold 64px system-ui, Arial, sans-serif';
  ctx.fillText(art.title || '', 40, 60);

  // Artist and year
  ctx.font = '36px system-ui, Arial, sans-serif';
  const byline = `${art.artist || ''}${art.year ? ', ' + art.year : ''}`;
  ctx.fillText(byline, 40, 150);

  // Materials
  ctx.font = '28px system-ui, Arial, sans-serif';
  ctx.fillText(art.materials || '', 40, 210);

  // Description (if space allows)
  if (art.description && art.description !== '—') {
    ctx.font = '24px system-ui, Arial, sans-serif';
    ctx.fillStyle = '#CCCCCC';
    const words = art.description.split(' ');
    let line = '';
    let y = 270;
    const maxWidth = 900;
    const lineHeight = 32;

    for (const word of words) {
      const testLine = line + word + ' ';
      const metrics = ctx.measureText(testLine);
      if (metrics.width > maxWidth && line !== '') {
        ctx.fillText(line, 40, y);
        line = word + ' ';
        y += lineHeight;
        if (y > 450) break; // Don't exceed canvas bounds
      } else {
        line = testLine;
      }
    }
    if (line && y <= 450) {
      ctx.fillText(line, 40, y);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}