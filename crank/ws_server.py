"""WebSocket server that streams world model frames to connected clients.

Each connected client receives one JSON-encoded VizFrame per message,
matching the protocol consumed by viz/visualizer-juicy.html (lines 2686-2707).

Usage:
    from crank.ws_server import serve
    asyncio.run(serve(model_args, port=8765, loop=True))
"""

import asyncio
import concurrent.futures
import json
import logging
import random
import time

import websockets

from models.checkpoint import load_model_from_checkpoint, CHARACTER_NAMES
from crank.agents import make_agent
from crank.match_runner import run_match_iter

logger = logging.getLogger(__name__)

# Thread pool for running synchronous inference off the event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Characters with animation support that appear frequently in training data.
# Subset of CHARACTER_NAMES keys — all have SVG ZIPs on the frontend.
TOURNAMENT_CHARS = [
    1,   # Fox
    2,   # Captain Falcon
    7,   # Sheik
    9,   # Peach
    10,  # Popo (Ice Climbers)
    12,  # Pikachu
    13,  # Samus
    15,  # Jigglypuff
    17,  # Luigi
    18,  # Marth
    21,  # Doc
    22,  # Falco
    25,  # Ganondorf
    26,  # Roy
]


async def serve(
    world_model_path: str,
    p0_spec: str,
    p1_spec: str,
    stage: int,
    p0_char: int,
    p1_char: int,
    max_frames: int = 600,
    device: str = "cpu",
    no_early_ko: bool = False,
    port: int = 8765,
    loop: bool = False,
):
    """Start the WebSocket server and stream matches to all connected clients.

    When a match ends, if loop=True, a new match starts automatically.
    Frames are paced at ~60fps (16.67ms per frame).
    """
    # Load model once at startup
    world_model, cfg, context_len, arch = load_model_from_checkpoint(
        world_model_path, device,
    )
    logger.info("World model: %s (%s), context=%d", world_model_path, arch, context_len)

    clients: set[websockets.WebSocketServerProtocol] = set()

    async def handler(ws: websockets.WebSocketServerProtocol):
        clients.add(ws)
        addr = ws.remote_address
        logger.info("Client connected: %s (%d total)", addr, len(clients))
        try:
            # Keep connection alive — we only send, never receive meaningful data
            async for _ in ws:
                pass
        finally:
            clients.discard(ws)
            logger.info("Client disconnected: %s (%d remaining)", addr, len(clients))

    async def broadcast(message: str):
        """Send a message to all connected clients."""
        if not clients:
            return
        # Send to all, remove any that error
        dead = set()
        for ws in clients:
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                dead.add(ws)
        clients.difference_update(dead)

    async def run_matches():
        """Run matches and broadcast frames. Waits for at least one client."""
        match_num = 0
        while True:
            # Wait for at least one client
            while not clients:
                await asyncio.sleep(0.1)

            match_num += 1

            # Pick random characters each match
            match_p0 = random.choice(TOURNAMENT_CHARS)
            match_p1 = random.choice(TOURNAMENT_CHARS)
            p0_name = CHARACTER_NAMES.get(match_p0, f"CHAR_{match_p0}")
            p1_name = CHARACTER_NAMES.get(match_p1, f"CHAR_{match_p1}")
            logger.info("Starting match #%d: %s vs %s", match_num, p0_name, p1_name)

            p0_agent = make_agent(p0_spec, player=0, cfg=cfg, device=device)
            p1_agent = make_agent(p1_spec, player=1, cfg=cfg, device=device)

            frame_gen = run_match_iter(
                world_model, cfg, p0_agent, p1_agent,
                stage, match_p0, match_p1, max_frames, device, no_early_ko,
            )

            frame_interval = 1.0 / 60.0  # 60fps
            frame_count = 0
            match_start = time.monotonic()
            loop = asyncio.get_event_loop()

            # Pull frames from the synchronous generator in a thread
            # so inference doesn't block the event loop (keeps WS alive).
            def next_frame():
                return next(frame_gen, None)

            while True:
                t_start = time.monotonic()

                # Run inference in thread pool to keep event loop responsive
                frame = await loop.run_in_executor(_executor, next_frame)
                if frame is None:
                    break

                msg = json.dumps(frame)
                await broadcast(msg)
                frame_count += 1

                # Pace at 60fps — include inference time in the budget
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Log FPS every 60 frames
                if frame_count % 60 == 0:
                    wall = time.monotonic() - match_start
                    fps = frame_count / wall
                    logger.info("Match #%d: frame %d (%.1f fps, %.1fms/frame)",
                                match_num, frame_count, fps, elapsed * 1000)

            wall = time.monotonic() - match_start
            fps = frame_count / wall if wall > 0 else 0
            logger.info("Match #%d ended: %d frames in %.1fs (%.1f fps)",
                        match_num, frame_count, wall, fps)

            if not loop:
                break

            # Brief pause between matches
            await asyncio.sleep(1.0)

    async with websockets.serve(handler, "0.0.0.0", port):
        logger.info("WebSocket server listening on ws://0.0.0.0:%d", port)
        await run_matches()

    logger.info("Server shutting down")
