# Rendering Architecture: Character Animation Overlay

How the visualizer renders Melee character animations, what the current approach can and can't do, and paths for expanding it.

---

## Current Pipeline: Pre-Baked SVG Silhouettes

The animation data comes from [SlippiLab](https://github.com/frankborden/slippilab) (MIT), which rasterized every frame of every Melee character animation into SVG path outlines. These are full 2D silhouettes — complete body traces in a 1000×1000 coordinate space, not skeletal rigs or sprites.

### Data Format

```
viz/zips/marth.zip          ← 26 character ZIPs, ~88MB total
  ├── Wait1.json            ← one JSON per animation
  ├── Run.json
  ├── AttackAirLw.json
  └── ...                   ← ~30-50 animations per character

Each JSON = array of SVG path strings (one per frame):
  ["M502.65 332h.15l3.5-.2...", "M502.5 331.9h.2...", ...]
```

Each path string is ~1200-1300 characters tracing the character's full outline for that animation frame. Marth's `Wait1.json` has 90 frames. It's a flipbook — no bones, no interpolation, no procedural movement.

### Frame Selection: Game State → SVG Path

```
action_state_id (0-399)
  ↓  ACTION_NAME_BY_ID[id]
action_name (e.g., "Wait")
  ↓  ANIMATION_REMAPS[name] (aliases: "Wait" → "Wait1", "AppealL" → "Appeal", etc.)
animation_name (e.g., "Wait1")
  ↓  animations[animName]
frames_array (90 SVG path strings)
  ↓  frames[Math.floor(state_age) % frames.length]
SVG path string
  ↓  new Path2D(pathStr)  ← cached by "extId:animName:frameIdx"
path2d object → ctx.fill(path2d)
```

**Frame references**: Some frames store `"frame20"` instead of a path string — an indirection meaning "reuse the SVG path from frame 20." Saves space for holds/freezes.

**Fallback chain**: If the animation doesn't exist for a character → fall back to `Wait1`. If `Wait1` is missing → fall back to capsule rendering.

### ID Mapping Chain

The world model outputs `character` as an internal Melee ID. The animation system needs an external ID to find the right ZIP:

```
character (internal, e.g., 18 = Marth)
  ↓  EXTERNAL_ID_BY_INTERNAL_ID[18] = 9
external ID (9)
  ↓  CHAR_ZIP[9] = "marth"
ZIP filename: zips/marth.zip
```

Tables ported from `nojohns/web/src/lib/meleeIds.ts` and `characterData.ts`.

### Transform Chain: SVG Space → Canvas Pixels

`drawCharacter()` applies six transforms in sequence:

```javascript
ctx.translate(px, py);              // ① Screen position (feet/origin)
ctx.scale(pxPerUnit, pxPerUnit);    // ② Game units → pixels (uniform)
ctx.scale(charScale, charScale);    // ③ Per-character sizing (0.55–1.30)
ctx.scale(facingDir, 1);            // ④ Horizontal flip (-1 or 1)
ctx.scale(0.1, -0.1);              // ⑤ SVG coords (0-1000) → game units + Y-flip
ctx.translate(-500, -500);          // ⑥ Center the 1000×1000 path space
ctx.fill(path2d);                   // ⑦ Render
```

| Step | What it does | Example (Marth, center stage, 1200px canvas) |
|------|-------------|----------------------------------------------|
| ① translate | Move to screen position | (600, 300) |
| ② pxPerUnit | Viewport scale | 3.5 px per game unit |
| ③ charScale | Character size factor | 1.05× for Marth |
| ④ facing | Mirror if facing left | 1 (facing right, no flip) |
| ⑤ scale(0.1, -0.1) | Shrink SVG space, flip Y | 1000 SVG units → 100 display units |
| ⑥ translate(-500, -500) | Center sprite on origin | Path point (500, 500) → (0, 0) |

**Known bug**: The Y-flip in step ⑤ double-inverts because `gameToScreen()` already flips Y. Characters render upside down. Fix: change to `scale(0.1, 0.1)`. See `viz/CODEX-character-anim-bugs.md`.

### Caching Layers

Two levels prevent redundant work:

1. **Animation cache** (`animCache: Map<externalId, animations>`) — ZIP decompressed and JSONs parsed once per character per session. ~500KB raw → ~50KB parsed per character.

2. **Path2D cache** (`path2dCache: Map<string, Path2D>`) — SVG string → Path2D parsed once per unique frame. Key format: `"9:Wait1:0"`. Grows over time (unbounded — a full match could reach ~50MB).

### Performance

- **First load**: ~100-300ms per character (network + fflate decompression + JSON.parse)
- **Subsequent frames**: sub-1ms (cache hit → fill)
- **Path2D.fill()**: GPU-accelerated in all modern browsers. 2 characters × 1 path each at 60fps is trivial.
- **The bottleneck** (if any) is the initial Path2D construction from complex SVG strings, but caching eliminates this after first encounter.

---

## What This Approach Can't Do

The pre-baked silhouette approach has hard limits:

- **No new animations** unless they exist in Melee and were extracted by SlippiLab
- **No new characters** beyond Melee's 26
- **No interpolation** between frames — it's a flipbook, so motion at low state_age increments looks choppy
- **No part-based rendering** — can't color the sword differently from the body, can't show hitboxes overlaid on limbs
- **No procedural variation** — every Marth looks identical, no costumes/colors
- **Frame data is large** — ~3MB per character uncompressed, because every frame is a full outline

---

## Library Assessment

### Motion (motion.dev)

[Motion](https://motion.dev/) (formerly Framer Motion) is a value interpolation engine — springs, keyframes, physics-based easing. ~18KB gzipped, MIT licensed.

**What it is**: An animation *timing* layer. It interpolates numbers, not pixels. You tell it "go from 0 to 100 with a spring," and it calls you back with intermediate values at 60fps.

**What it could do for us**:
- Smooth position transitions when the model's predicted (x, y) jumps between frames
- Spring-based screen shake on hits (replace the current random-offset approach)
- Eased opacity/scale transitions when switching between capsule and character mode
- Timeline sequencing for combo counter pop-ins, stock loss animations, UI juice
- `animate(obj.property, target, { type: 'spring', stiffness: 300 })` can directly mutate properties read by our render loop

**What it can't do**: Anything related to rendering. It doesn't touch the canvas. It's complementary to whatever renderer we use.

### Three.js (threejs.org)

[Three.js](https://threejs.org/) is a WebGL/WebGPU 3D renderer. ~600KB, MIT licensed.

**2D capability**: `OrthographicCamera` gives flat projection. `PlaneGeometry` + `MeshBasicMaterial` renders textured quads. Sprite sheet animation via UV offset cycling (GPU handles frame switching — a few bytes of data per frame, not a full path parse).

**Animation system**: `AnimationMixer` supports skeletal animation (bone hierarchies via `SkinnedMesh`), morph targets, and property keyframing. Overkill for flipbook animation, perfect for skeletal rigs.

**For our use case**: Could replace the entire rendering pipeline, but it's a heavyweight choice for what's currently a 2D problem. The right move if we want 2.5D (parallax stages, particle effects, depth) or if we're adding skeletal characters.

**PixiJS** ([pixijs.com](https://pixijs.com/)) is the proportionate alternative for pure 2D: WebGL-accelerated sprite rendering, automatic batching, purpose-built for this. Same GPU performance, simpler API, smaller footprint.

### Comparison

| | Current (Canvas Path2D) | + Motion | PixiJS | Three.js |
|---|---|---|---|---|
| **Renderer** | Canvas 2D | Canvas 2D | WebGL 2D | WebGL 2D/3D |
| **Animation data** | SVG path flipbook | SVG path flipbook | Sprite sheets | Skeletal or sprites |
| **New characters** | Need SVG outlines | Need SVG outlines | Need sprite sheets | Need rigs or sprites |
| **Performance** | Fine for 2 characters | Same + smoother juice | Excellent at scale | Excellent + 3D |
| **Interpolation** | None (frame snap) | Value tweening | Frame blending possible | Full skeletal blend |
| **Complexity** | Minimal | Low addition | Medium rewrite | Heavy rewrite |
| **Bundle** | 0 (native) | ~18KB | ~200KB | ~600KB |

---

## Expansion Paths

### Path 1: Extend Pre-Baked Silhouettes (minimal effort)

Stay with the current system. Add Motion for juice timing. Accept the 26-character limit.

- **Add Motion** for screen shake springs, transition easing, UI animation timing
- **Optimize Path2D cache** with LRU eviction (cap at ~20MB)
- **Add color variation** by rendering the same Path2D multiple times with different fills/strokes (e.g., red team / blue team tint)
- **Hitbox overlay**: render a second, simpler Path2D for active hitboxes (would need separate hitbox data, not in SlippiLab)

Good for: the current visualizer use case where we're rendering Melee match data.

### Path 2: Sprite Sheet Migration (medium effort)

Rasterize the SVG paths into texture atlases. Render with PixiJS or Three.js. Opens the door to non-Melee characters.

- **Convert existing SVGs** to PNG sprite sheets at fixed resolution (e.g., 128×128 per frame)
- **Render via WebGL** — GPU handles frame cycling as UV offset, no Path2D parsing
- **Custom characters** become "provide a sprite sheet" — can be hand-drawn, AI-generated, or sourced from other games' sprite databases
- **PixiJS** is the natural fit here — purpose-built for 2D sprite rendering

Good for: scaling beyond Melee's roster while keeping the 2D aesthetic.

### Path 3: Skeletal Animation (high effort, highest ceiling)

Characters defined as rigs (skeleton + mesh), not frame collections. New animations = new bone keyframes. New characters = new mesh on same rig.

- **Spine** ([esotericsoftware.com](http://esotericsoftware.com/)) is the industry standard for 2D skeletal animation. Exports to canvas, WebGL, Three.js. A single rigged character with 20 animations might be 500KB vs 3MB of pre-baked paths.
- **DragonBones** is the open-source alternative
- **Three.js** supports skeletal animation natively via `SkinnedMesh` + `AnimationMixer`
- **The unlock**: animation blending (smooth transitions between states), procedural adjustments (lean into movement direction), and dramatically smaller file sizes per character
- **The cost**: every character needs to be rigged. Can't port SlippiLab data into this format — it's a fundamentally different representation

Good for: building toward "bring your own character" / custom character generation.

### The Hybrid (recommended direction)

Keep SlippiLab SVG silhouettes for Melee's 26 characters — they're authentic and they work. Add a skeletal system for new characters. Motion drives the timing layer for both. The world model doesn't care — it outputs `action_state`, `x`, `y`, `facing`, `state_age`, and the rendering layer maps those numbers to visuals however it wants.

```
World Model Output (structured state)
  ├── Melee character? → SlippiLab SVG flipbook (existing)
  └── Custom character? → Spine/skeletal rig (new)
        ↓
  [same action_state → animation mapping]
  [same transform chain]
  [same canvas/WebGL output]
```

This preserves the project's core architectural insight: **the model outputs physics, rendering is decoupled**. The same match could be rendered as SVG silhouettes, pixel sprites, 3D models, or ASCII art. The physics don't change.

---

## Relation to the World Model

The rendering layer consumes exactly the fields the model outputs:

| Model output | Rendering use |
|---|---|
| `character` | Which ZIP / sprite sheet / rig to load |
| `action_state` | Which animation to play |
| `state_age` | Which frame of that animation |
| `x, y` | Screen position |
| `facing` | Horizontal flip |
| `percent`, `stocks` | HUD display |
| `shield_strength` | Shield bubble size |
| `hitlag` | Flash effect |
| `speed_*` | Velocity arrows |

The rendering system makes zero gameplay decisions. It's a pure function: `render(state) → pixels`. This separation is what makes the "model IS the world" thesis work — the world's physics are defined by the model, and any rendering approach is just a viewport into that world.

---

## Source Files

| File | Role |
|---|---|
| `viz/visualizer-juicy.html` | Production visualizer (juice effects, replay loading) |
| `viz/zips/*.zip` | SlippiLab animation data (26 characters, ~88MB) |
| `viz/CODEX-character-anim-bugs.md` | Open bugs (Y-flip, replay loader) |
| `rnd-2026/.../visualizer.html` | Prototype with character animation code (reference impl) |
| `nojohns/.../meleeIds.ts` | ID mapping tables (source of truth) |
| `nojohns/.../characterData.ts` | Scale factors, animation remaps |
| `nojohns/.../animationCache.ts` | Reference ZIP loading / caching |
| `nojohns/.../MeleeViewer.tsx` | Reference SVG renderer |
