#!/usr/bin/env python3
"""Live demo server: two agents playing in the world model, streamed to the viewer.

Loads a world model + policy checkpoint(s), picks a seed game, and runs
an agent-vs-agent match. Frames stream over WebSocket to viewer.html.

Usage:
    # Policy vs policy (same model, P0 and P1):
    .venv/bin/python -m worldmodel.scripts.serve_demo \
        --world-model worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
        --p0 policy:worldmodel/checkpoints/policy-22k-v2/best.pt \
        --p1 policy:worldmodel/checkpoints/policy-22k-v2/best.pt \
        --seed-game ~/claude-projects/nojohns-training/data/parsed-v2/games/<md5>

    # Policy vs scripted:
    .venv/bin/python -m worldmodel.scripts.serve_demo \
        --world-model worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
        --p0 policy:worldmodel/checkpoints/policy-22k-v2/best.pt \
        --p1 hold-forward

    # Random seed game from a directory:
    .venv/bin/python -m worldmodel.scripts.serve_demo \
        --world-model worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
        --p0 policy:worldmodel/checkpoints/policy-22k-v2/best.pt \
        --p1 random \
        --seed-dir ~/claude-projects/nojohns-training/data/parsed-v2/games

Opens http://localhost:8765 in the viewer.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

import torch
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.dataset import _encode_game
from data.parse import load_game
from models.encoding import EncodingConfig
from scripts.generate_demo import (
    STAGE_GEOMETRY,
    _build_predicted_player,
    _resolve_character_name,
    compute_summary,
    load_model_from_checkpoint,
)
from scripts.play_match import make_agent, play_match
from scripts.rollout import clamp_frame, decode_frame

logger = logging.getLogger(__name__)

VIEWER_PATH = Path(__file__).resolve().parent.parent / "viewer.html"


def pick_seed_game(seed_game: str | None, seed_dir: str | None) -> Path:
    """Pick a seed game — explicit path or random from directory."""
    if seed_game:
        return Path(seed_game).expanduser()

    if seed_dir:
        games_dir = Path(seed_dir).expanduser()
        games = [g for g in games_dir.iterdir() if g.is_file() and not g.name.startswith(".")]
        if not games:
            raise FileNotFoundError(f"No games found in {games_dir}")
        chosen = random.choice(games)
        logger.info("Picked random seed: %s", chosen.name)
        return chosen

    raise ValueError("Must provide --seed-game or --seed-dir")


def generate_match_data(
    world_model: torch.nn.Module,
    cfg: EncodingConfig,
    context_len: int,
    arch: str,
    seed_game_path: Path,
    p0_spec: str,
    p1_spec: str,
    max_frames: int,
    device: str,
    game_harness: bool = False,
) -> dict:
    """Generate a full match and return viewer-compatible JSON."""
    game = load_game(str(seed_game_path))
    float_data, int_data = _encode_game(game, cfg)
    logger.info("Seed game: %d frames, stage=%d", game.num_frames, game.stage)

    stage_id = game.stage
    p0_char_id = int(game.p0.character[0])
    p1_char_id = int(game.p1.character[0])

    stage_geo = STAGE_GEOMETRY.get(stage_id, {
        "name": f"Stage {stage_id}",
        "ground_y": 0, "ground_x_range": [-85, 85],
        "platforms": [],
        "blast_zones": {"left": -240, "right": 240, "top": 200, "bottom": -140},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    })

    p0_agent = make_agent(p0_spec, player=0, float_data=float_data, cfg=cfg, device=device)
    p1_agent = make_agent(p1_spec, player=1, float_data=float_data, cfg=cfg, device=device)

    frames = play_match(
        world_model, float_data, int_data, cfg,
        p0_agent, p1_agent,
        max_frames=max_frames,
        device=device,
        no_early_ko=False,
        stage_geometry=stage_geo if game_harness else None,
        game_harness=game_harness,
    )

    summary = compute_summary(frames, "agent-vs-agent")

    return {
        "meta": {
            "mode": "agent-vs-agent",
            "model_name": "live-demo",
            "arch": arch,
            "context_len": context_len,
            "total_frames": len(frames),
            "seed_frames": context_len,
            "p0_agent": p0_spec,
            "p1_agent": p1_spec,
            "stage": {"id": stage_id, "name": stage_geo["name"]},
            "characters": {
                "p0": {"id": p0_char_id, "name": _resolve_character_name(p0_char_id)},
                "p1": {"id": p1_char_id, "name": _resolve_character_name(p1_char_id)},
            },
        },
        "stage_geometry": stage_geo,
        "frames": frames,
        "summary": summary,
    }


async def handle_viewer(request):
    """Serve the viewer HTML with WebSocket support injected."""
    html = VIEWER_PATH.read_text()

    # Inject WebSocket connection code before the closing </script> tag
    ws_script = """

// ===== WebSocket live mode =====
let ws = null;
let wsConnected = false;

function connectWebSocket() {
  const wsUrl = `ws://${location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsConnected = true;
    console.log('Connected to demo server');
    // Request a new match
    ws.send(JSON.stringify({type: 'start'}));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'match') {
      // Full match data — same format as file drop
      demoData = msg.data;
      initViewer();
      play();
    } else if (msg.type === 'status') {
      console.log('Server:', msg.message);
    }
  };

  ws.onclose = () => {
    wsConnected = false;
    console.log('Disconnected from demo server');
    // Reconnect after a short delay
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
}

// Add restart button functionality
function restartMatch() {
  if (ws && wsConnected) {
    stop();
    currentFrame = 0;
    ws.send(JSON.stringify({type: 'start'}));
  }
}

// Auto-connect on page load (only when served from the demo server)
if (location.port === '8765' || location.search.includes('live=1')) {
  // Hide drop zone, show a "connecting" message
  dropZone.querySelector('label').innerHTML = '<div style="font-size:40px;margin-bottom:12px;">&#9889;</div><div>Connecting to demo server...</div>';

  connectWebSocket();
}
"""

    html = html.replace("</script>", ws_script + "\n</script>")

    # Add a "New Match" button (replaces "Load new file" in live mode)
    restart_btn = '<button id="restart-btn" style="display:none;position:fixed;top:12px;left:12px;background:var(--surface2);border:1px solid var(--border);color:var(--text-dim);border-radius:4px;padding:6px 12px;cursor:pointer;font-family:inherit;font-size:12px;z-index:10;" onclick="restartMatch()">New Match</button>'
    html = html.replace('<button id="new-file-btn"', restart_btn + '\n<button id="new-file-btn"')

    # Show restart button in live mode
    show_restart = """
// Show restart button in live mode
if (location.port === '8765' || location.search.includes('live=1')) {
  const origInit = initViewer;
  initViewer = function() {
    origInit();
    document.getElementById('restart-btn').style.display = 'block';
  };
}
"""
    html = html.replace("</script>", show_restart + "\n</script>")

    return web.Response(text=html, content_type="text/html")


async def handle_ws(request):
    """WebSocket endpoint — generates and streams matches."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket client connected")

    app = request.app

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)

            if data.get("type") == "start":
                await ws.send_json({"type": "status", "message": "Generating match..."})

                try:
                    # Pick a new seed game each time
                    seed_path = pick_seed_game(
                        app["args"].seed_game,
                        app["args"].seed_dir,
                    )

                    # Generate the match (blocking — run in executor to not freeze the server)
                    match_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: generate_match_data(
                            world_model=app["world_model"],
                            cfg=app["cfg"],
                            context_len=app["context_len"],
                            arch=app["arch"],
                            seed_game_path=seed_path,
                            p0_spec=app["args"].p0,
                            p1_spec=app["args"].p1,
                            max_frames=app["args"].max_frames,
                            device=app["args"].device,
                            game_harness=app["args"].game_harness,
                        ),
                    )

                    await ws.send_json({"type": "match", "data": match_data})
                    logger.info(
                        "Sent match: %d frames (%d seed + %d agent)",
                        len(match_data["frames"]),
                        app["context_len"],
                        len(match_data["frames"]) - app["context_len"],
                    )

                except Exception as e:
                    logger.exception("Match generation failed")
                    await ws.send_json({"type": "status", "message": f"Error: {e}"})

        elif msg.type == web.WSMsgType.ERROR:
            logger.error("WebSocket error: %s", ws.exception())

    logger.info("WebSocket client disconnected")
    return ws


def main():
    parser = argparse.ArgumentParser(description="Live demo server")
    parser.add_argument("--world-model", required=True, help="Path to world model checkpoint")
    parser.add_argument("--seed-game", help="Path to a specific seed game")
    parser.add_argument("--seed-dir", help="Directory of parsed games (picks randomly)")
    parser.add_argument("--p0", default="random", help="P0 agent spec")
    parser.add_argument("--p1", default="random", help="P1 agent spec")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames per match")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--game-harness", action="store_true", default=True,
                        help="Enable game rules harness (default: on)")
    parser.add_argument("--no-game-harness", action="store_false", dest="game_harness",
                        help="Disable game rules harness")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate seed source
    if not args.seed_game and not args.seed_dir:
        parser.error("Must provide --seed-game or --seed-dir")

    # Load world model once at startup
    logger.info("Loading world model from %s...", args.world_model)
    world_model, cfg, context_len, arch = load_model_from_checkpoint(
        args.world_model, args.device,
    )
    logger.info("World model ready: %s (%s), K=%d", args.world_model, arch, context_len)

    # Build the app
    app = web.Application()
    app["world_model"] = world_model
    app["cfg"] = cfg
    app["context_len"] = context_len
    app["arch"] = arch
    app["args"] = args

    app.router.add_get("/", handle_viewer)
    app.router.add_get("/ws", handle_ws)

    logger.info("Starting demo server at http://localhost:%d", args.port)
    logger.info("P0: %s, P1: %s", args.p0, args.p1)
    web.run_app(app, host="localhost", port=args.port, print=None)


if __name__ == "__main__":
    main()
