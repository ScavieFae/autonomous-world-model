/**
 * Controller input handling — keyboard → Melee controller mapping.
 *
 * Maps keyboard inputs to GameCube controller state for the world model.
 * Two control schemes: player 1 (WASD) and player 2 (arrow keys).
 */

// ── GCC button bitmask ──────────────────────────────────────────────────────

export const GCC_A = 0x01;
export const GCC_B = 0x02;
export const GCC_X = 0x04;
export const GCC_Y = 0x08;
export const GCC_Z = 0x10;
export const GCC_START = 0x20;
export const GCC_DLEFT = 0x40;
export const GCC_DRIGHT = 0x80;

// Extended buttons
export const GCC_DUP = 0x01;
export const GCC_DDOWN = 0x02;
export const GCC_L = 0x04;
export const GCC_R = 0x08;

// ── Controller state ────────────────────────────────────────────────────────

export interface ControllerInput {
  stickX: number; // -128 to 127
  stickY: number; // -128 to 127
  cStickX: number;
  cStickY: number;
  triggerL: number; // 0 to 255
  triggerR: number;
  buttons: number; // Bitmask
  buttonsExt: number;
}

export function defaultInput(): ControllerInput {
  return {
    stickX: 0,
    stickY: 0,
    cStickX: 0,
    cStickY: 0,
    triggerL: 0,
    triggerR: 0,
    buttons: 0,
    buttonsExt: 0,
  };
}

// ── Keyboard mapping ────────────────────────────────────────────────────────

/** Default keyboard → controller mapping for player 1 (WASD layout) */
export const P1_KEYMAP = {
  // Main stick
  KeyA: { axis: "stickX" as const, value: -80 },
  KeyD: { axis: "stickX" as const, value: 80 },
  KeyW: { axis: "stickY" as const, value: 80 },
  KeyS: { axis: "stickY" as const, value: -80 },

  // Buttons
  KeyJ: { button: GCC_A },   // A (attack)
  KeyK: { button: GCC_B },   // B (special)
  KeyU: { button: GCC_X },   // X (jump)
  KeyI: { button: GCC_Y },   // Y (jump)
  KeyL: { button: GCC_Z },   // Z (grab)

  // C-stick (smash attacks)
  ArrowLeft: { axis: "cStickX" as const, value: -80 },
  ArrowRight: { axis: "cStickX" as const, value: 80 },
  ArrowUp: { axis: "cStickY" as const, value: 80 },
  ArrowDown: { axis: "cStickY" as const, value: -80 },

  // Triggers
  KeyQ: { trigger: "triggerL" as const, value: 255 }, // L (shield)
  KeyE: { trigger: "triggerR" as const, value: 255 }, // R (shield)
};

/** Default keyboard → controller mapping for player 2 (numpad layout) */
export const P2_KEYMAP = {
  Numpad4: { axis: "stickX" as const, value: -80 },
  Numpad6: { axis: "stickX" as const, value: 80 },
  Numpad8: { axis: "stickY" as const, value: 80 },
  Numpad5: { axis: "stickY" as const, value: -80 },

  Numpad1: { button: GCC_A },
  Numpad2: { button: GCC_B },
  Numpad7: { button: GCC_X },
  Numpad9: { button: GCC_Y },
  Numpad3: { button: GCC_Z },

  Numpad0: { trigger: "triggerL" as const, value: 255 },
  NumpadDecimal: { trigger: "triggerR" as const, value: 255 },
};

// ── Input state manager ─────────────────────────────────────────────────────

type KeyMap = Record<string, { axis?: "stickX" | "stickY" | "cStickX" | "cStickY"; value?: number; button?: number; trigger?: "triggerL" | "triggerR" }>;

export class InputManager {
  private keysDown = new Set<string>();
  private keymap: KeyMap;
  private _onInputChange?: () => void;

  constructor(keymap: KeyMap = P1_KEYMAP) {
    this.keymap = keymap;
  }

  /** Start listening for keyboard events. Call stop() to clean up. */
  start(element: HTMLElement | Window = window): () => void {
    const onKeyDown = (e: Event) => {
      const ke = e as KeyboardEvent;
      if (this.keymap[ke.code]) {
        ke.preventDefault();
        this.keysDown.add(ke.code);
        this._onInputChange?.();
      }
    };

    const onKeyUp = (e: Event) => {
      const ke = e as KeyboardEvent;
      this.keysDown.delete(ke.code);
      this._onInputChange?.();
    };

    element.addEventListener("keydown", onKeyDown);
    element.addEventListener("keyup", onKeyUp);

    return () => {
      element.removeEventListener("keydown", onKeyDown);
      element.removeEventListener("keyup", onKeyUp);
      this.keysDown.clear();
    };
  }

  /** Set callback for input state changes. */
  onInputChange(cb: () => void) {
    this._onInputChange = cb;
  }

  /** Get current controller state from pressed keys. */
  getInput(): ControllerInput {
    const input = defaultInput();

    for (const code of this.keysDown) {
      const mapping = this.keymap[code];
      if (!mapping) continue;

      if (mapping.axis && mapping.value !== undefined) {
        input[mapping.axis] = clampI8(input[mapping.axis] + mapping.value);
      }
      if (mapping.button) {
        input.buttons |= mapping.button;
      }
      if (mapping.trigger && mapping.value !== undefined) {
        input[mapping.trigger] = Math.min(255, input[mapping.trigger] + mapping.value);
      }
    }

    return input;
  }
}

function clampI8(v: number): number {
  return Math.max(-128, Math.min(127, Math.round(v)));
}

// ── Gamepad support (future) ────────────────────────────────────────────────

/**
 * Read Gamepad API state and convert to ControllerInput.
 * For GCC adapters (Mayflash, official), axis mapping is:
 *   Axes 0,1: main stick (X, Y)
 *   Axes 2,3: c-stick (X, Y)
 *   Axes 4,5: triggers (L, R)
 */
export function readGamepad(gamepadIndex: number = 0): ControllerInput | null {
  if (typeof navigator === "undefined" || !navigator.getGamepads) return null;

  const gp = navigator.getGamepads()[gamepadIndex];
  if (!gp) return null;

  return {
    stickX: clampI8(gp.axes[0] * 80),
    stickY: clampI8(-gp.axes[1] * 80), // Y is inverted on gamepads
    cStickX: clampI8((gp.axes[2] ?? 0) * 80),
    cStickY: clampI8(-(gp.axes[3] ?? 0) * 80),
    triggerL: Math.round((gp.axes[4] ?? -1 + 1) / 2 * 255),
    triggerR: Math.round((gp.axes[5] ?? -1 + 1) / 2 * 255),
    buttons:
      (gp.buttons[0]?.pressed ? GCC_A : 0) |
      (gp.buttons[1]?.pressed ? GCC_B : 0) |
      (gp.buttons[2]?.pressed ? GCC_X : 0) |
      (gp.buttons[3]?.pressed ? GCC_Y : 0) |
      (gp.buttons[4]?.pressed ? GCC_Z : 0) |
      (gp.buttons[9]?.pressed ? GCC_START : 0),
    buttonsExt:
      (gp.buttons[12]?.pressed ? GCC_DUP : 0) |
      (gp.buttons[13]?.pressed ? GCC_DDOWN : 0) |
      (gp.buttons[6]?.pressed ? GCC_L : 0) |
      (gp.buttons[7]?.pressed ? GCC_R : 0),
  };
}
