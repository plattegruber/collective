// Use global THREE set by index.html
const { THREE } = window;

export async function buildOverlay(art) {
  const group = new THREE.Group();

  // Get panel dimensions for proper scaling
  const { w, h } = art.overlay.panel;
  
  // Create refined glow effect using SDF-based shader
  const color = new THREE.Color(0xbed9ff);
  const uniforms = {
    u_time: { value: 0 },
    u_color: { value: color.toArray().slice(0, 3) },
    u_maxAlpha: { value: 0.7 },
    u_inset: { value: 0.06 },
    u_radius: { value: 0.08 },
    u_feather: { value: 0.08 },
    u_breatheHz: { value: 0.55 },
    u_shimmerHz: { value: 0.22 },
    u_shimmerAmp: { value: 0.6 },
  };

  const vertexShader = /* glsl */`
    varying vec2 vUv;
    void main() { 
      vUv = uv; 
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); 
    }
  `;

  const fragmentShader = /* glsl */`
    precision mediump float;
    varying vec2 vUv;

    uniform vec3  u_color;
    uniform float u_time, u_maxAlpha;
    uniform float u_inset, u_radius, u_feather;
    uniform float u_breatheHz, u_shimmerHz, u_shimmerAmp;

    // SDF for rounded rectangle in [0,1]^2 (after inset)
    float sdRoundRect(vec2 p, vec2 b, float r){
      vec2 q = abs(p) - (b - vec2(r));
      return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    // Soft pulse 0..1
    float pulse(float t) { return 0.5 + 0.5 * sin(6.2831853 * t); }

    void main() {
      // Centered coords, apply inset so we don't chase raw quad edges
      vec2 uv = mix(vec2(0.0), vec2(1.0), vUv);
      vec2 c = uv - 0.5;
      vec2 size = vec2(0.5 - u_inset);      // half-extent after inset
      float d = sdRoundRect(c, size, u_radius);

      // Feathered inner mask (negative d is inside the rect)
      float edge = smoothstep(u_feather, 0.0, -d);     // fades to 1 toward center
      float innerGlow = smoothstep(0.0, -u_feather, d); // slight inner bloom near edge

      // Breathing alpha + subtle inner bloom toward edges
      float breath = 0.75 + 0.25 * pulse(u_time * u_breatheHz);
      float base = mix(edge * 0.6, 1.0, innerGlow);    // richer near the frame edge

      // Shimmer: a slow diagonal lightfall that travels across the rect interior
      float diag = dot(normalize(vec2(0.7, 0.7)), c);  // -~0.7..~0.7
      float sweep = 1.0 - smoothstep(0.0, 0.25, abs(fract(diag + u_time * u_shimmerHz) - 0.5));
      float shimmer = sweep * edge * u_shimmerAmp;

      float intensity = clamp(base * breath + shimmer, 0.0, 1.0);
      vec3  col = u_color * intensity;
      float alpha = intensity * u_maxAlpha;

      gl_FragColor = vec4(col, alpha);
    }
  `;

  const glowMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
  });

  // Create plane geometry and mesh
  const plane = new THREE.PlaneGeometry(1, 1, 1, 1);
  const glowMesh = new THREE.Mesh(plane, glowMat);
  
  // Scale to match artwork dimensions exactly
  glowMesh.scale.set(w, h, 1);
  glowMesh.position.z = 0.01;
  
  group.add(glowMesh);
  group.userData = { uniforms };

  return group;
}

