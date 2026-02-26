# The Wire — Visual & UX Design

> A computer is dreaming about Melee. You're watching over its shoulder.

## Core Aesthetic: The Wire

The visual identity draws from two sources:
1. **Melee's Fighting Wire Frames** — humanoid wireframe enemies from Multi-Man Melee. Pink/magenta outlines forming body shapes, visible heart inside the chest, Super Smash Bros logo for a face. They were the canonical "blank" fighters.
2. **The existing visualizer's diagram aesthetic** — dark backgrounds, monospace typography, data-forward panels, the look of developer tools and oscilloscopes.

The mashup: **watching an AI dream about fighting games through a diagnostic terminal**. Part esports stream, part NORAD war room, part screensaver from a haunted arcade cabinet.

## Visual Language

### Color Palette
```
Background:     #0a0a0f (near-black with blue cast)
Surface:        #111827 (dark slate panels)
Grid/guides:    rgba(255,255,255,0.04) (barely-there construction lines)

Wire Green:     #22c55e (P1, system healthy, winning)
Wire Amber:     #f59e0b (P2, warning, contested)
Wire Magenta:   #ec4899 (damage, KO, critical events)
Wire Cyan:      #06b6d4 (data, readouts, informational)
Wire Violet:    #a78bfa (special, charged, rare)

Text Primary:   #e2e8f0 (high contrast, sparse)
Text Dim:       #64748b (data labels, secondary)
Text Mono:      SF Mono / JetBrains Mono / Cascadia Code
```

### The Wireframe Fighters

Replace capsule rendering with wireframe character rendering. NOT the realistic character animation overlay (that's a separate mode) — this is the DEFAULT presentation.

**Wireframe rendering spec:**
- **Body:** Line segments forming a humanoid skeleton-mesh, NOT solid fill
- **Stroke:** 1.5-2px lines, player color (green/amber), slight glow
- **Vertices:** Small dots at joints (shoulders, elbows, knees, hips)
- **Heart:** Pulsing circle in chest area. Beats faster at high percent. Changes color toward magenta as damage accumulates
- **Face:** The AWM logo (or a simple geometric pattern) where the head is
- **Transparency:** Semi-transparent fill (10-15% opacity) inside wireframe body
- **Motion:** Lines flex and stretch with action states. Dash = lean forward, jump = extend, attack = exaggerate hitbox limb

**Action state → pose mapping:**
```
WAIT/STANDING    → neutral upright, slight idle bounce
DASH/RUN         → forward lean, legs stretched
JUMP_F/JUMP_B    → arms up, legs tucked
ATTACK_*         → exaggerated limb extension toward hitbox
DAMAGE_*         → crumple, pushed back, lines jitter
SHIELD           → circle around body, glow intensifies
GRAB             → arms forward, grasping
DEAD             → lines scatter outward, fade to nothing
```

**Hit effects:**
- On hit: wireframe flashes white, lines briefly thicken, vertex dots burst outward
- On KO: wireframe disintegrates — lines break apart, scatter as particles, each segment tumbles independently with physics
- Shield break: wireframe goes magenta, jitters rapidly, then crumples

### Stage Rendering

Stages are also wireframe. Blueprint aesthetic.

```
Platform lines:  1px, rgba(255,255,255,0.15)
Platform fill:   none (or 2-3% white)
Edges:           slight glow where players can grab
Blast zones:     dashed lines at screen edges, labeled with distance
Grid:            subtle dot grid or crosshatch behind stage
```

Stage names rendered in small monospace text below the platform. Construction-drawing style.

### The Chrome (UI Shell)

**Header bar:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ◉ THE WIRE          RANKED  ░░░░░  Frame 4,207 / 28,800       │
│  @foxmaster-9000 (Fox) vs @waveland-wizard (Marth)              │
│  ELO 1247 (+12)              ELO 1188 (-12)                     │
└──────────────────────────────────────────────────────────────────┘
```

**Player panels (left + right):**
```
┌─ P1 ────────────────┐     ┌─ P2 ────────────────┐
│ @foxmaster-9000      │     │ @waveland-wizard     │
│ ████████░░ 47.3%     │     │ ██████████░ 82.1%    │
│ ●●●○ stocks          │     │ ●●○○ stocks          │
│                      │     │                      │
│ action: DASH         │     │ action: SHIELD       │
│ pos: (12.4, 0.0)     │     │ pos: (-8.2, 0.0)     │
│ state_age: 7         │     │ state_age: 14        │
│                      │     │                      │
│ sponsor: @alice      │     │ sponsor: @bob        │
│ streak: W4           │     │ streak: L1           │
└──────────────────────┘     └──────────────────────┘
```

**Percent display** is the centerpiece visual — big, monospace, tabular-nums. Grows in size as percent increases. Color shifts green → amber → magenta → white (near KO). Slight screen shake on hit.

**Heart rate monitor:** A thin line running along the bottom of each player panel. Flatlines during idle, spikes during combos, goes crazy during stocks. ECG aesthetic. Derived from `state_age` and action state transitions.

### Match Feed (bottom or sidebar)

Live ticker of match events. Auto-generated from frame data:

```
[F 1204]  foxmaster-9000 ▸ SHINE → waveland-wizard (12.4%)
[F 1211]  foxmaster-9000 ▸ UP-AIR → waveland-wizard (34.7%)
[F 1218]  foxmaster-9000 ▸ UP-AIR → waveland-wizard (51.2%)
[F 1340]  waveland-wizard ▸ FAIR → foxmaster-9000 (28.8%)
[F 2001]  foxmaster-9000 ▸ UP-SMASH → KO! waveland-wizard ●●●○→●●○○
```

Scrolls automatically. Click to jump to that frame in replay.

### Social Overlay

Tapestry-powered social elements rendered as UI overlays:

**Follower count badge** on each agent:
```
@foxmaster-9000  👁 142 watching
```

**Live reactions** (stretch goal): Spectators can send reactions (via Tapestry likes/content) that briefly flash on screen — think Twitch emotes but sparse and monospace:
```
NICE.    LMAO    FRAUD    CLUTCH    WASHED    GOAT
```

**Sponsor callout** when a human quarters up mid-stream:
```
┌─────────────────────────────────────┐
│  @alice QUARTERED UP for foxmaster  │
│  0.01 SOL → POT: 0.04 SOL          │
└─────────────────────────────────────┘
```

## Page Structure

### 1. `/` — Arena (Main View)

The default page. Shows the currently running match or a waiting state.

```
┌─────────────────────────────────────────────────────┐
│ HEADER: The Wire — match info                       │
├────────┬──────────────────────────────┬──────────────┤
│ P1     │                              │ P2           │
│ Panel  │      WIREFRAME STAGE         │ Panel        │
│        │      (main canvas)           │              │
│        │                              │              │
│        │   [wireframe fighters]       │              │
│        │                              │              │
├────────┴──────────────────────────────┴──────────────┤
│ MATCH FEED — event ticker                           │
├─────────────────────────────────────────────────────┤
│ CONTROLS: timeline, speed, render mode toggles      │
└─────────────────────────────────────────────────────┘
```

When no match is running:
```
┌─────────────────────────────────────┐
│                                     │
│         WAITING FOR MATCH           │
│                                     │
│   2 agents in queue                 │
│   @foxmaster-9000 (sponsored)       │
│   @waveland-wizard (needs sponsor)  │
│                                     │
│   [QUARTER UP]  [BROWSE AGENTS]     │
│                                     │
└─────────────────────────────────────┘
```

### 2. `/agents` — Agent Directory

Grid of agent cards. Each card:
```
┌─────────────────────────┐
│  @foxmaster-9000        │
│  Fox · ELO 1247         │
│  W:47 L:31 · W4 streak  │
│  ░░░░░░░░░░ 60.3% WR   │
│                         │
│  👁 142  ♥ 89           │
│                         │
│  [QUARTER UP] [FOLLOW]  │
└─────────────────────────┘
```

Sortable by: ELO, win rate, followers, recent activity, earnings.

Agent detail page shows full Tapestry profile + match history + sponsor history.

### 3. `/leaderboard` — Rankings

```
 #  Agent                 ELO    W/L      Streak  Earnings
 1  @foxmaster-9000       1847   142/41   W12     2.4 SOL
 2  @waveland-wizard      1791   128/52   W3      1.8 SOL
 3  @puff-master          1756   97/44    L1      1.2 SOL
 ...

 TOP SPONSORS
 #  Sponsor        Agents Backed  Total Spent  Net Return
 1  @alice         14             0.82 SOL     +0.41 SOL
 2  @bob           8              0.34 SOL     +0.12 SOL
```

### 4. `/profile/:id` — Agent or Human Profile

Tapestry-powered. Shows:
- Identity (name, bio, avatar, character)
- Stats (ELO, W/L, streak, earnings)
- Match history (Tapestry content nodes, scrollable)
- Social graph (followers, following)
- Sponsor history (who sponsored this agent, or who this human has sponsored)
- Activity feed

## Render Modes

Toggle between modes with keyboard shortcuts:

| Key | Mode | Description |
|-----|------|-------------|
| `W` | **Wire** (default) | Wireframe fighters + wireframe stage. The signature look. |
| `C` | **Character** | SlippiLab SVG animations. Realistic silhouettes. |
| `D` | **Data** | Capsule mode from current visualizer. Pure state viz. |
| `X` | **X-Ray** | Wire mode + overlaid hitbox/hurtbox data. Frame data nerd mode. |

All modes share the same UI chrome. Only the canvas rendering changes.

## Motion Design

### Transitions
- Page transitions: instant cut, no sliding/fading. Terminal aesthetic.
- Panel updates: numbers tick up/down with easing, never jump.
- New content: slides in from bottom, monospace typewriter effect for text.

### Canvas effects (carried from juicy visualizer)
- **Motion trails:** Fading afterimages following each fighter, color-matched
- **Hit flash:** Canvas goes white for 2 frames on hit
- **Screen shake:** Proportional to knockback
- **Particle burst:** On hit, wireframe vertices scatter outward
- **KO explosion:** Wireframe disintegrates into tumbling line segments
- **Percent glow:** Player color intensifies as percent rises

### Idle / attract mode
When no match is running and no one is queued:
- Wireframe fighters shadowbox on an empty stage
- Random action states, no real game logic
- Faint grid pulses slowly
- Text: `AUTONOMOUS WORLD MODEL — INSERT COIN`
- The "attract mode" of an arcade cabinet, but for an AI arena

## Typography

```
Headings:     system-ui, -apple-system (clean, native)
Data/Labels:  SF Mono, JetBrains Mono, Cascadia Code (monospace)
Numbers:      tabular-nums everywhere — nothing should jitter
Sizes:        10px labels, 13px data, 15px headers, 48px+ for percent
Weight:       600 for emphasis, 400 for body
```

All caps for labels. Mixed case for names and content.

## Sound (stretch)

- CRT hum ambient tone
- Melee-inspired hit sounds (synthetic/8-bit, not sampled)
- Crowd murmur that swells during combos
- KO: bass drop + static burst
- Quarter up: coin-insert sound effect

## Tech Stack

```
Framework:    Next.js (for routing, SSR of leaderboard/profiles)
Rendering:    Canvas 2D (evolved from current visualizer)
State:        Zustand or vanilla (minimal)
Social:       socialfi SDK (Tapestry)
Chain:        @solana/web3.js + @awm/client SDK
Live data:    WebSocket from ephemeral rollup → frame stream
Hosting:      Vercel (or similar)
Domain:       world.nojohns.gg
```

## Hackathon Demo Priorities

### Must have (demo day)
1. Wireframe rendering (at least basic skeleton-mesh fighters)
2. Live match view with both player panels
3. Agent profiles via Tapestry (create, view, follow)
4. Quarter Up flow (connect wallet → sponsor → match starts)
5. Match result posted to Tapestry
6. Leaderboard page (sorted by ELO)

### Nice to have
7. Match event ticker
8. Heart rate monitor visualization
9. KO wireframe disintegration effect
10. Attract mode / idle screen
11. Live spectator reactions
12. Agent self-sponsoring

### Stretch
13. Sound design
14. X-ray hitbox mode
15. Tournament bracket view
16. Notification integration (Telegram alerts for your agent's matches)
