const{THREE:c}=window;async function z(r){const t=new c.Group,{w:a,h:d}=r.overlay.panel,n=new c.Color(12507647),o={u_time:{value:0},u_color:{value:n.toArray().slice(0,3)},u_maxAlpha:{value:.7},u_inset:{value:.06},u_radius:{value:.08},u_feather:{value:.08},u_breatheHz:{value:.55},u_shimmerHz:{value:.22},u_shimmerAmp:{value:.6}},m=`
    varying vec2 vUv;
    void main() { 
      vUv = uv; 
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); 
    }
  `,h=`
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
  `,v=new c.ShaderMaterial({uniforms:o,vertexShader:m,fragmentShader:h,transparent:!0,depthWrite:!1,depthTest:!1,blending:c.AdditiveBlending}),p=new c.PlaneGeometry(1,1,1,1),u=new c.Mesh(p,v);return u.scale.set(a,d,1),u.position.z=.01,t.add(u),t.userData={uniforms:o},t}const{THREE:x,MindARThree:C}=window;async function A({map:r,content:t,versions:a}){const d=new C({container:document.body,imageTargetSrc:`/collective/targets/${a.mindFile}`,maxTrack:a.maxTrack,filterMinCF:.01,filterBeta:5,warmupTolerance:3,missTolerance:3}),{renderer:n,scene:o,camera:m}=d;n.setClearColor(0,0),n.autoClear=!0;const h=new x.HemisphereLight(16777215,4473924,1);o.add(h);const v=document.getElementById("artwork-info"),p=document.getElementById("artwork-title"),u=document.getElementById("artwork-artist"),E=document.getElementById("artwork-materials"),T=document.getElementById("artwork-description");for(const{index:e,artId:i}of r.targets){const f=d.addAnchor(e),s=await z(t[i]);s.visible=!1,f.group.add(s);const l=t[i];f.onTargetFound=()=>{s.visible=!0,s.traverse(g=>{g.visible=!0}),p.textContent=l.title||"",u.textContent=`${l.artist||""}${l.year?", "+l.year:""}`,E.textContent=l.materials||"",T.textContent=l.description||"",v.classList.add("visible")},f.onTargetLost=()=>{s.visible=!1,s.traverse(g=>{g.visible=!1}),v.classList.remove("visible")}}window.addEventListener("resize",()=>{n.setSize(window.innerWidth,window.innerHeight),m.aspect=window.innerWidth/window.innerHeight,m.updateProjectionMatrix()}),await d.start();const y=document.querySelectorAll("video"),_=document.querySelectorAll("canvas");if(y.length>0){const e=y[0];setTimeout(()=>{e.style.setProperty("z-index","0","important"),e.style.setProperty("visibility","visible","important"),e.style.setProperty("display","block","important")},100)}if(_.length>0){const e=_[0];e.style.setProperty("z-index","1","important"),e.style.setProperty("background","transparent","important")}const k=new x.Clock,b=()=>{requestAnimationFrame(b);const e=k.getElapsedTime();o.traverse(i=>{i.userData?.uniforms&&i.visible&&(i.userData.uniforms.u_time.value=e)}),n.render(o,m)};b()}const w={targetsMap:"v1",artContent:"v1",mindFile:"gallery-v1.mind",maxTrack:2};(async()=>{try{const[r,t]=await Promise.all([fetch(`/collective/targets/targets-map.${w.targetsMap}.json`).then(a=>a.json()),fetch(`/collective/data/art-content.${w.artContent}.json`).then(a=>a.json())]);await A({map:r,content:t,versions:w}),document.getElementById("loading").style.display="none"}catch(r){console.error("Failed to initialize AR tracker:",r),document.getElementById("loading").textContent="Failed to load. Please refresh."}})();export{w as VERSIONS};
