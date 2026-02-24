# Task 001: Visualizer JSON Loader

## Goal

The visualizer currently runs on procedurally generated mock data. Add the ability to load real world model output from a JSON file, so we can visualize actual model predictions.

## What to change

**File:** `viz/visualizer.html`

## Requirements

### 1. JSON file loading

Add a drag-and-drop zone and file picker button to load a `.json` file of frames. The drop target should be the canvas area — drop a file anywhere on the stage and it loads.

Also add a file picker button near the data source dropdown in the header.

### 2. Expected JSON format

The file contains an array of frame objects. Each frame:

```json
{
  "players": [
    {
      "x": -30.2,
      "y": 0.0,
      "percent": 42,
      "shield_strength": 60.0,
      "speed_air_x": 0.0,
      "speed_y": 0.0,
      "speed_ground_x": 1.8,
      "speed_attack_x": 0.0,
      "speed_attack_y": 0.0,
      "state_age": 12,
      "hitlag": 0,
      "stocks": 4,
      "facing": 1,
      "on_ground": 1,
      "action_state": 20,
      "jumps_left": 2,
      "character": 18
    },
    { "...same shape for P2..." }
  ],
  "stage": 31
}
```

Top-level file structure is just the array:
```json
[
  { "players": [...], "stage": 31 },
  { "players": [...], "stage": 31 },
  ...
]
```

### 3. Validation on load

When a file is loaded:
- Parse JSON, check it's an array of objects
- Check first frame has `players` array with 2 entries
- Check first player has required fields: `x`, `y`, `action_state`
- If validation fails, show an error message on the canvas (not an alert)
- If valid, replace the current frame data and reset playback to frame 0

### 4. Keep mock data as fallback

The three existing mock scenarios (neutral, combo, edgeguard) should still work. The data source dropdown should gain a fourth option "Loaded file" that appears after a successful file load, and auto-selects.

### 5. Header tag update

When mock data is active, show the "Mock Data" tag (already exists). When a loaded file is active, change it to "Model Output" with an amber color instead of green.

### 6. Generate a test fixture

Create `viz/fixtures/test-sequence.json` — a 120-frame (2 second) test file with two players. Generate it with a simple script or by hand. It should have:
- Both players on Final Destination (stage 31)
- Player 1: Marth (character 18), starts at x=-30, moves right, throws a ftilt (action_state 44) around frame 40
- Player 2: Fox (character 1), starts at x=30, gets hit around frame 45, knockback visible in speed_attack_x/y
- Plausible values — positions in [-100, 100], velocities in [-5, 5], percent increasing after hits

This fixture proves the loader works end-to-end.

## What NOT to do

- Don't restructure the existing code into modules/files. Keep it as a single self-contained HTML file.
- Don't add a build system or dependencies.
- Don't change the rendering logic or visual style.
- Don't add network/WebSocket loading — just local file loading for now.

## Done when

1. Opening `viz/visualizer.html` still works with mock data by default
2. Dragging `viz/fixtures/test-sequence.json` onto the canvas loads and plays the sequence
3. File picker button in the header also works
4. Data source dropdown shows "Loaded file" after load and auto-selects it
5. Switching back to a mock scenario works after loading a file
6. Invalid JSON shows an error on the canvas, doesn't crash
