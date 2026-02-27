#!/usr/bin/env python3
"""CLI entry point for the offchain match runner.

Mode A (standalone): two agents fight inside the world model, output JSON.
    python -m crank.main \
        --world-model checkpoints/world-model.pt \
        --p0 policy:checkpoints/policy.pt \
        --p1 policy:checkpoints/policy.pt \
        --stage 2 --p0-char 2 --p1-char 2 \
        --max-frames 600 --output match.json

Mode B (Solana crank): reads/writes ER accounts.
    python -m crank.main \
        --world-model checkpoints/world-model.pt \
        --policy checkpoints/policy.pt \
        --session <pubkey> --rpc <er-endpoint>

Mode C (WebSocket server): stream live matches to browser clients.
    python -m crank.main \
        --world-model checkpoints/world-model.pt \
        --p0 policy:checkpoints/policy.pt \
        --p1 policy:checkpoints/policy.pt \
        --serve --port 8765 --loop
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import torch

from models.checkpoint import load_model_from_checkpoint
from crank.agents import make_agent
from crank.match_runner import run_match

logger = logging.getLogger(__name__)


def run_serve(args):
    """Mode C: WebSocket server — stream matches to connected clients."""
    from crank.ws_server import serve

    asyncio.run(serve(
        world_model_path=args.world_model,
        p0_spec=args.p0,
        p1_spec=args.p1,
        stage=args.stage,
        p0_char=args.p0_char,
        p1_char=args.p1_char,
        max_frames=args.max_frames,
        device=args.device,
        no_early_ko=args.no_early_ko,
        port=args.port,
        loop=args.loop,
    ))


def run_standalone(args):
    """Mode A: standalone match — two agents, output JSON."""
    # Load world model
    world_model, cfg, context_len, arch = load_model_from_checkpoint(
        args.world_model, args.device,
    )
    logger.info("World model: %s (%s), context=%d", args.world_model, arch, context_len)

    # Create agents
    p0_agent = make_agent(args.p0, player=0, cfg=cfg, device=args.device)
    p1_agent = make_agent(args.p1, player=1, cfg=cfg, device=args.device)
    logger.info("P0: %s, P1: %s", args.p0, args.p1)

    # Run match
    result = run_match(
        world_model, cfg, p0_agent, p1_agent,
        stage=args.stage,
        p0_char=args.p0_char,
        p1_char=args.p1_char,
        max_frames=args.max_frames,
        device=args.device,
        no_early_ko=args.no_early_ko,
    )

    # Add model info to meta
    result["meta"]["model_checkpoint"] = str(args.world_model)
    result["meta"]["arch"] = arch
    result["meta"]["context_len"] = context_len
    result["meta"]["p0_agent"] = args.p0
    result["meta"]["p1_agent"] = args.p1

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    total_frames = result["meta"]["total_frames"]
    logger.info("Wrote %s (%.1f MB, %d frames)", args.output, size_mb, total_frames)

    # Summary
    if total_frames > context_len:
        last = result["frames"][-1]
        p0 = last["players"][0]
        p1 = last["players"][1]
        logger.info(
            "Final: P0 stocks=%.0f pct=%.1f%% | P1 stocks=%.0f pct=%.1f%%",
            p0["stocks"], p0["percent"], p1["stocks"], p1["percent"],
        )


async def run_crank(args):
    """Mode B: Solana crank — read ER accounts, run inference, write back.

    This is the prototype crank loop. It polls the InputBuffer for both
    players' inputs, runs inference, and writes the results back.
    """
    from crank.solana_bridge import (
        read_session_state, read_input_buffer, write_session_state,
        STATUS_ACTIVE,
    )
    from crank.state_convert import (
        session_to_tensors, controller_to_tensor, tensors_to_session,
    )

    # Load world model
    world_model, cfg, context_len, arch = load_model_from_checkpoint(
        args.world_model, args.device,
    )
    logger.info("World model: %s (%s), context=%d", args.world_model, arch, context_len)

    # Create policy agents
    p0_agent = make_agent(f"policy:{args.policy}", player=0, cfg=cfg, device=args.device)
    p1_agent = make_agent(f"policy:{args.policy}", player=1, cfg=cfg, device=args.device)

    logger.info("Crank started for session %s", args.session)
    logger.info("RPC: %s", args.rpc)

    # Main crank loop
    context_buffer_f = []  # sliding window of float frames
    context_buffer_i = []  # sliding window of int frames
    K = context_len

    while True:
        # Read session state
        session = await read_session_state(args.rpc, args.session)
        if session is None:
            logger.error("Session not found, exiting")
            break
        if session.status != STATUS_ACTIVE:
            logger.info("Session status=%d, not active. Exiting.", session.status)
            break

        # Convert to tensors and maintain context buffer
        float_frame, int_frame = session_to_tensors(session, cfg)
        context_buffer_f.append(float_frame)
        context_buffer_i.append(int_frame)

        # Wait until we have K frames of context
        if len(context_buffer_f) < K:
            await asyncio.sleep(0.016)  # ~60fps
            continue

        # Keep only last K frames
        context_buffer_f = context_buffer_f[-K:]
        context_buffer_i = context_buffer_i[-K:]

        ctx_f = torch.stack(context_buffer_f)
        ctx_i = torch.stack(context_buffer_i)

        # Get controller inputs from policies
        p0_ctrl = p0_agent.get_controller(ctx_f, ctx_i, cfg, session.frame)
        p1_ctrl = p1_agent.get_controller(ctx_f, ctx_i, cfg, session.frame)
        next_ctrl = torch.cat([p0_ctrl, p1_ctrl]).unsqueeze(0).to(args.device)

        # Run world model
        with torch.no_grad():
            preds = world_model(
                ctx_f.unsqueeze(0).to(args.device),
                ctx_i.unsqueeze(0).to(args.device),
                next_ctrl,
            )

        # TODO: decode predictions into new SessionState and write back
        # For now just log
        logger.info("Frame %d: inference done", session.frame)

        await asyncio.sleep(0.001)  # Yield to event loop


def main():
    parser = argparse.ArgumentParser(
        description="Offchain match runner for the autonomous world model",
    )
    parser.add_argument(
        "--world-model", required=True,
        help="Path to world model checkpoint (.pt)",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("-v", "--verbose", action="store_true")

    # Mode A: standalone
    standalone = parser.add_argument_group("standalone mode")
    standalone.add_argument("--p0", default="hold-forward", help="P0 agent spec")
    standalone.add_argument("--p1", default="hold-forward", help="P1 agent spec")
    standalone.add_argument("--stage", type=int, default=32, help="Stage ID (default: FD=32)")
    standalone.add_argument("--p0-char", type=int, default=2, help="P0 character ID (default: Captain Falcon=2)")
    standalone.add_argument("--p1-char", type=int, default=2, help="P1 character ID (default: Captain Falcon=2)")
    standalone.add_argument("--max-frames", type=int, default=600, help="Max frames")
    standalone.add_argument("--output", default="match.json", help="Output JSON path")
    standalone.add_argument("--no-early-ko", action="store_true", help="Don't stop on KO")

    # Mode B: Solana crank
    crank = parser.add_argument_group("crank mode")
    crank.add_argument("--session", help="Session account pubkey")
    crank.add_argument("--rpc", help="Solana RPC endpoint (ER)")
    crank.add_argument("--policy", help="Policy checkpoint for both players")

    # Mode C: WebSocket server
    ws = parser.add_argument_group("serve mode")
    ws.add_argument("--serve", action="store_true", help="Start WebSocket server")
    ws.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    ws.add_argument("--loop", action="store_true", help="Loop matches continuously")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.serve:
        # Mode C: WebSocket server
        run_serve(args)
    elif args.session and args.rpc:
        # Mode B: Solana crank
        if not args.policy:
            parser.error("--policy is required for crank mode")
        asyncio.run(run_crank(args))
    else:
        # Mode A: standalone
        run_standalone(args)


if __name__ == "__main__":
    main()
