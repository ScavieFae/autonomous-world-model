"""Offchain match runner — runs world model inference, writes results.

Two modes:
  A. Standalone: two PolicyAgents fight inside the world model, output JSON.
  B. Solana crank: reads/writes ER accounts, runs inference loop.
"""
