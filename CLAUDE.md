# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- `npm run dev` - Start development server on port 3000 (HTTPS required for camera access on mobile)
- `npm run build` - Build for production (outputs to `dist/`)  
- `npm run serve` - Preview production build locally

### Dependencies
- `npm install` - Install all dependencies (Vite, Three.js)

## Architecture

This is a **MindAR Three.js Gallery** - an AR application for overlaying artwork information using image tracking.

### Core Architecture Pattern

The app uses a **versioned content system** for hot-swapping gallery data without rebuilding:

1. **Version Control Center** (`src/main.js`): Central `VERSIONS` object controls which data files to load
2. **Multi-target Tracking**: Single `.mind` file tracks multiple artworks simultaneously  
3. **Decoupled Content**: JSON files define artwork info separately from tracking data

### Key Files & Responsibilities

- **`src/main.js`**: Bootstrap and version configuration. Contains the `VERSIONS` object that controls which data files are loaded
- **`src/tracker.js`**: MindAR setup, camera management, anchor creation, and target detection events
- **`src/overlays.js`**: Three.js overlay generation (panels, text textures, optional thumbnails)
- **`index.html`**: CDN imports (MindAR, Three.js) and app initialization
- **`targets/gallery-v*.mind`**: Compiled image targets for MindAR recognition
- **`targets/targets-map.v*.json`**: Maps MindAR target indices to stable artwork IDs
- **`data/art-content.v*.json`**: Complete artwork metadata and overlay specifications

### Data Flow

1. `main.js` loads target map and content based on version config
2. `tracker.js` creates MindAR instance and anchors for each target
3. `overlays.js` generates Three.js overlay groups from artwork data
4. Target detection triggers overlay show/hide with fade animations

### Version Management System

**When to bump versions:**
- New artwork images → Increment `mindFile` and `targetsMap` versions  
- Text/layout/audio changes → Only increment `artContent` version
- Always update corresponding version in `VERSIONS` object in `src/main.js`

**Example version update:**
```javascript
export const VERSIONS = {
  targetsMap: 'v2',              // ← Updated for new targets
  artContent: 'v1',              // ← Keep same if only images changed
  mindFile: 'gallery-v2.mind',   // ← Updated for new .mind file
  maxTrack: 2                    // ← Max simultaneous tracking
};
```

### External Dependencies (CDN)

- **MindAR**: `mind-ar@1.2.5` (image tracking)
- **Three.js**: `three@0.150.0` (3D rendering)

Both loaded via CDN in `index.html` and attached to `window` object for module access.

### Mobile-First Considerations

- Requires HTTPS for camera access
- Development server configured for `0.0.0.0:3000` for mobile testing
- Optimized for iOS Safari and Android Chrome
- `maxTrack: 2` prevents performance issues on mobile devices