# Codex Task: Fix Character Animation Rendering Bugs

## File to modify
`viz/visualizer-juicy.html`

## Context
Character animation rendering was just added — toggling with `C` key loads SlippiLab SVG animation data from `viz/zips/` and renders Melee character silhouettes instead of abstract capsules. The animation code was initially prototyped in a copy at `rnd-2026/projects/autonomous-world-model/visualizer.html` — that version has the full working implementation (ID mappings, animation cache, `drawCharacter()`, toggle button, keyboard shortcut) but needs to be ported into `visualizer-juicy.html` which is the real file with all the juice effects (trails, particles, shake, etc).

## Reference implementation
The character animation code lives in: `/Users/mattiefairchild/claude-projects/rnd-2026/projects/autonomous-world-model/visualizer.html`

Port the following sections into `visualizer-juicy.html`:
1. **fflate CDN script** — `<script src="https://cdn.jsdelivr.net/npm/fflate@0.8.2/umd/index.js">` in `<head>`
2. **CSS** — `.ctrl-btn.loading` pulse animation
3. **ID mapping tables** — `EXTERNAL_ID_BY_INTERNAL_ID`, `ACTION_NAME_BY_ID` (341 entries), `ANIMATION_REMAPS`, `CHAR_SCALE`, `CHAR_ZIP`
4. **Animation cache** — `loadCharAnimations()`, `resolveAnimFrame()`, `getPath2D()`, `path2dCache`
5. **`drawCharacter()`** — SVG Path2D renderer with transform chain
6. **`drawCapsule()`** — extracted capsule drawing (refactored out of inline code)
7. **Rendering conditional** — `if (renderMode === 'character') { drawCharacter || fallback to drawCapsule }`
8. **Toggle button** — `<button id="btnRenderMode">Capsule</button>` in `.btn-row`, `C` keyboard shortcut, `toggleRenderMode()` function

## Bug 1: Characters are upside down

**Cause:** Double Y-inversion. The `drawCharacter()` transform chain uses `scale(0.1, -0.1)` to flip SVG coordinates into game units. But `gameToScreen()` already inverts Y: `sy = ((viewTop - gy) / (viewTop - viewBottom)) * canvasH`. So the character gets flipped twice.

**Fix:** In `drawCharacter()`, change:
```javascript
ctx.scale(0.1, -0.1);
```
to:
```javascript
ctx.scale(0.1, 0.1);
```

May also need to adjust the `translate(-500, -500)` offset — the SVG path space is 0-1000 on both axes, centered at (500, 500). With the Y-flip removed, the vertical positioning might shift. Test with the Marth/Fox neutral scenario — characters' feet should align with the stage platform (y=0 in game coords).

The nojohns reference renderer (`MeleeViewer.tsx` line 110) uses:
```
scale(0.1 -0.1) translate(-500 -500)
```
...but that's in an SVG `<g transform="scale(1 -1)">` context where Y is already flipped globally. Our canvas `gameToScreen()` does the equivalent flip, so we should NOT re-flip.

## Bug 2: Verify replay match loader still works

`visualizer-juicy.html` has a replay data loading mechanism (file picker / JSON fetch) that the prototype version didn't have. After porting the character animation code, verify that:
- Loading replay JSON data still works
- Character mode renders correctly with real replay data (not just mock scenarios)
- The `character` field from replay data correctly maps through `EXTERNAL_ID_BY_INTERNAL_ID` → ZIP filename → animation

## Animation data
ZIPs are at `viz/zips/*.zip` (26 files, ~88MB). Already gitignored. Each ZIP contains JSON files (one per animation), each JSON is an array of SVG path strings (one per frame).

## How to test
1. Serve from repo root or `viz/`: `cd viz && python3 -m http.server 8420`
2. Open `http://localhost:8420/visualizer-juicy.html`
3. Capsule mode should work unchanged (all juice effects intact)
4. Press `C` — characters should load and render RIGHT-SIDE UP, feet on the platform
5. All action states should animate (dash, attack, damage, shield, etc.)
6. Press `C` again — instant switch back to capsules
7. Load a replay JSON — both capsule and character modes work

## Source reference files (read-only)
- `/Users/mattiefairchild/claude-projects/nojohns/web/src/components/viewer/MeleeViewer.tsx` — reference renderer
- `/Users/mattiefairchild/claude-projects/nojohns/web/src/lib/animationCache.ts` — reference ZIP loading
- `/Users/mattiefairchild/claude-projects/nojohns/web/src/lib/characterData.ts` — scale factors, remaps
- `/Users/mattiefairchild/claude-projects/rnd-2026/projects/autonomous-world-model/visualizer.html` — prototype with all animation code implemented
