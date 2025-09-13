# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- `npm run dev` - Start development server on port 3000 (HTTPS required for camera access on mobile)
- `npm run build` - Build for production (outputs to `dist/`)
- `npm run serve` - Preview production build locally

### Dependencies
- `npm install` - Install all dependencies (Vite, Three.js, Tailwind CSS)

## Architecture

This is an **ONNX-powered AR Gallery** - a web application that uses computer vision to recognize artworks and display information overlays.

### Core Architecture Pattern

The app is a **single-page application** with embedded JavaScript that uses ONNX Runtime for artwork detection:

1. **ONNX Model Pipeline**: Custom-trained detector model recognizes multiple artworks simultaneously
2. **Real-time Detection**: Continuous video processing with configurable confidence thresholds (70%)
3. **Overlay System**: Dynamic information overlays triggered by artwork detection

### Key Files & Responsibilities

- **`index.html`**: Complete application with embedded JavaScript - camera setup, ONNX model loading, detection loop, and UI overlays
- **`public/models/detector/model.onnx`**: Trained ONNX model for artwork recognition
- **`public/models/detector/labels.json`**: Maps model class IDs to artwork identifiers (e.g., `{"1": "2d:byrons_painting"}`)
- **`public/data/art-content.v1.json`**: Artwork metadata including title, artist, year, materials, description
- **`public/model-manifest.json`**: Model versioning and path configuration
- **`vite.config.js`**: Build configuration with base path `/collective/` and mobile-optimized dev server

### Data Flow

1. App loads ONNX model and artwork content in parallel during initialization
2. Camera stream feeds into real-time detection pipeline (320x320 input size)
3. Model outputs bounding boxes, confidence scores, and class labels
4. Detections above 70% confidence threshold trigger overlay updates
5. Artwork information displays based on label mapping to content database

### Detection System

**Model Architecture:**
- Input: 320x320 RGB images (letterboxed and scaled from camera feed)
- Output: Bounding boxes, confidence scores, class labels
- Runtime: ONNX Runtime Web with WebGL/WebAssembly fallback
- Performance: Optimized for mobile browsers

**Detection Pipeline:**
```javascript
// Frame processing pipeline
processFrame() → decodeOutputs() → drawBoxes() → updateArtworkOverlay()
```

**Confidence Threshold:** 70% (configurable via `CONFIDENCE_THRESHOLD` constant)

### Content Management

**Adding New Artworks:**
1. Train new model with additional artwork images
2. Update `public/models/detector/labels.json` with new class mappings
3. Add artwork metadata to `public/data/art-content.v1.json`
4. Update `public/model-manifest.json` with new model version/hash
5. Deploy updated model and content files

**Data Structure:**
```javascript
// labels.json - maps model classes to artwork IDs
{"1": "2d:byrons_painting", "2": "2d:horse_swing"}

// art-content.v1.json - artwork metadata
{
  "2d:byrons_painting": {
    "title": "Untitled",
    "artist": "Byron Anway",
    "year": 2022,
    "materials": "Oil on canvas",
    "description": "A gift to Andrea."
  }
}
```

### Deployment Configuration

**Build Setup:**
- Static deployment via Vite build system
- Base path: `/collective/` (configured in `vite.config.js`)
- Output directory: `dist/`
- Deployment target: Static hosting (project.toml indicates Paketo static buildpack)

**Mobile Considerations:**
- HTTPS required for camera access
- Development server on `0.0.0.0:3000` for mobile testing
- Optimized for iOS Safari and Android Chrome
- Responsive overlay system with touch-friendly interactions