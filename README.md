# MindAR Three.js Gallery

AR overlays for artworks using MindAR image tracking with Three.js overlays. This app recognizes multiple artworks from a single compiled `.mind` file and renders dynamic overlays from JSON data.

## Features

- 📱 **Multi-target tracking**: Single `.mind` file tracks multiple artworks simultaneously
- 🎯 **Dynamic overlays**: Three.js overlays with title, artist, materials, and optional thumbnails  
- 🔄 **No-code updates**: Add new artworks by updating JSON files and recompiling `.mind` file
- 📱 **Mobile optimized**: Smooth performance on iOS Safari and Android Chrome
- 🚀 **Static deployment**: Ready for Netlify or Fly.io deployment

## Project Structure

```
mindar-three-gallery/
├── index.html                    # Entry point with camera canvas
├── package.json                  # Dependencies and scripts
├── vite.config.js                # Build configuration  
├── public/
│   └── assets/                  # Static assets (images, audio)
├── targets/
│   ├── gallery-v1.mind           # Multi-target recognition file
│   └── targets-map.v1.json       # Maps target index → artId
├── data/
│   └── art-content.v1.json       # Artwork metadata & overlay layout
└── src/
    ├── main.js                   # App bootstrap & version config
    ├── tracker.js                # MindAR setup & target management
    └── overlays.js               # Three.js overlay generation
```

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash  
   npm run dev
   ```

3. **Open http://localhost:3000** on your mobile device

4. **Point camera at your tracked images** to see AR overlays

## Data Structure

### Target Mapping (`targets/targets-map.v1.json`)
Maps MindAR target indices to stable artwork IDs:
```json
{
  "version": "v1",
  "targets": [
    { "index": 0, "artId": "EDW-2022-BlueBoat" },
    { "index": 1, "artId": "HOK-2019-Sunset" }
  ]
}
```

### Artwork Content (`data/art-content.v1.json`)
Complete artwork information and overlay specifications:
```json
{
  "EDW-2022-BlueBoat": {
    "title": "Blue Boat",
    "artist": "Erik Daniel White", 
    "year": 2022,
    "materials": "Oil on canvas",
    "description": "A serene maritime scene...",
    "overlay": {
      "panel": { "w": 1.0, "h": 0.55, "z": 0.01 }
    },
    "assets": {
      "image": "/public/assets/edw/blueboat.jpg",
      "audioGuide": "/public/assets/edw/blueboat.mp3"
    }
  }
}
```

## Adding New Artworks

### When to Recompile vs. Edit JSON

- **New artwork image** → Recompile `.mind` file
- **Fix text/layout/audio** → Edit `art-content.*.json` only (no rebuild needed)

### Step-by-Step Update Process

1. **Add image to MindAR compiler** and generate new multi-target file:
   ```
   gallery-v1.mind → gallery-v2.mind
   ```

2. **Update target mapping** in `targets/targets-map.v2.json`:
   ```json
   {
     "version": "v2", 
     "targets": [
       { "index": 0, "artId": "EDW-2022-BlueBoat" },
       { "index": 1, "artId": "HOK-2019-Sunset" },
       { "index": 2, "artId": "NEW-2024-Masterpiece" }
     ]
   }
   ```

3. **Add artwork content** to `data/art-content.v1.json` (or bump to v2):
   ```json
   {
     "NEW-2024-Masterpiece": {
       "title": "New Masterpiece",
       "artist": "Famous Artist",
       // ... rest of content
     }
   }
   ```

4. **Update version constants** in `src/main.js`:
   ```javascript
   export const VERSIONS = {
     targetsMap: 'v2',        // ← Updated
     artContent: 'v1', 
     mindFile: 'gallery-v2.mind',  // ← Updated
     maxTrack: 2
   };
   ```

5. **Build and deploy**:
   ```bash
   npm run build
   # Deploy dist/ folder to Netlify/Fly.io
   ```

## Deployment

### Netlify (Recommended)
1. Connect your GitHub repo or drag-and-drop `dist/` folder
2. Build command: `npm run build`
3. Publish directory: `dist`
4. Deploy! Netlify handles caching headers automatically.

### Fly.io  
1. Install Fly CLI: `brew install flyctl`
2. Create app: `fly launch`
3. Deploy: `fly deploy`

## Development

- **Dev server**: `npm run dev` (with hot reload)
- **Build**: `npm run build` (outputs to `dist/`)
- **Preview build**: `npm run serve`

## Technical Notes

- **MindAR version**: 1.2.5 (UMD build from CDN)
- **Three.js version**: 0.165.0 (loaded from CDN)  
- **Max simultaneous tracking**: 2 targets (configurable via `maxTrack`)
- **Supported formats**: Works on modern mobile browsers with WebRTC support
- **Performance**: Optimized for mobile with `maxTrack: 2` and efficient overlay rendering

## Troubleshooting

- **Camera not starting**: Ensure HTTPS and camera permissions granted
- **Targets not detected**: Check lighting and try moving closer to artwork
- **Performance issues**: Reduce `maxTrack` value or simplify overlay content
- **Build errors**: Clear `node_modules` and reinstall: `rm -rf node_modules package-lock.json && npm install`

## Version Management

Files use version suffixes for cache busting:
- `targets-map.v1.json`, `art-content.v1.json`, `gallery-v1.mind`
- Bump versions in `src/main.js` when content changes
- Old files can be kept for rollback capability