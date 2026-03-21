#!/usr/bin/env python3
"""
AWM docs prebuild — generates doc pages from project state.

DeepWiki-inspired patterns:
  1. File-relevance mapping: manifest.json declares which source files each doc
     page covers. Only regenerate pages whose sources changed.
  2. Two-phase generation: structure (what pages exist) is defined by the manifest
     and nav. Content is generated per-page from source files.
  3. Cache-first: skip pages whose source files haven't changed.

Run before `zensical build`:
    python scripts/docs_prebuild.py
"""

import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
RUN_CARDS_DIR = DOCS_DIR / "run-cards"
BASE_BUILDS_DIR = DOCS_DIR / "base-builds"
EXPERIMENTS_DIR = DOCS_DIR / "experiments"


def parse_yaml_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter from markdown. Minimal parser — no PyYAML needed."""
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    fm = {}
    for line in match.group(1).strip().split("\n"):
        line = line.strip()
        if ":" not in line or line.startswith("#"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Handle arrays like [e017a, e016]
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        elif val in ("null", "~", ""):
            val = None
        elif val in ("true", "True"):
            val = True
        elif val in ("false", "False"):
            val = False
        else:
            val = val.strip("'\"")
        fm[key] = val
    return fm


def parse_yaml_comments(text: str) -> dict:
    """Parse a YAML file with comments (like base build files). Extract key: value pairs."""
    fm = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        elif val in ("null", "~", ""):
            val = None
        else:
            val = val.strip("'\"")
        fm[key] = val
    return fm


def _node_key(card):
    """Unique Mermaid node key from filename stem (handles duplicate IDs like e018b)."""
    return card.get("filename", "").replace(".md", "").replace("-", "_")


def _build_experiment_tree(cards):
    """Build a Mermaid flowchart of experiment lineage from built_on relationships."""
    # Only include cards that have built_on or are referenced by built_on
    graphed = [c for c in cards if c.get("built_on") is not None or c.get("base_build")]
    if not graphed:
        return []

    status_styles = {
        "kept": "fill:#2e7d32,color:#fff",
        "running": "fill:#1565c0,color:#fff",
        "proposed": "fill:#f9a825,color:#000",
        "discarded": "fill:#616161,color:#fff",
        "cancelled": "fill:#424242,color:#999",
    }

    lines = [
        "## Experiment Tree",
        "",
        "```mermaid",
        "flowchart TD",
    ]

    # Build lookup: card ID → list of node keys (for edge targets)
    id_to_keys = {}
    for c in graphed:
        cid = c.get("id", "?")
        id_to_keys.setdefault(cid, []).append(_node_key(c))

    # Collect all referenced base builds
    base_builds = set()
    for c in graphed:
        bb = c.get("base_build")
        if bb:
            base_builds.add(bb)

    # Add base build nodes
    for bb in sorted(base_builds):
        lines.append(f'    {bb}["{bb}"]')

    # Add experiment nodes with RC in label
    for c in graphed:
        nk = _node_key(c)
        cid = c.get("id", "?")
        rc = c.get("rollout_coherence")
        status = c.get("status", "")
        label = cid
        if rc:
            label += f"\\nRC {rc}"
        if status == "kept":
            lines.append(f'    {nk}(["{label}"])')
        elif status in ("discarded", "cancelled"):
            lines.append(f'    {nk}["{label}"]')
        else:
            lines.append(f'    {nk}("{label}")')

    # Add edges
    for c in graphed:
        nk = _node_key(c)
        built_on = c.get("built_on", [])
        if isinstance(built_on, list) and built_on:
            for parent_id in built_on:
                # Connect from first node with that ID (usually only one)
                parent_keys = id_to_keys.get(parent_id, [])
                if parent_keys:
                    lines.append(f"    {parent_keys[0]} --> {nk}")
                else:
                    # Parent might be a pre-schema card not in the graph
                    lines.append(f"    {parent_id} --> {nk}")
        elif isinstance(built_on, list) and not built_on:
            # Empty list = root experiment, connect to base build
            bb = c.get("base_build")
            if bb:
                lines.append(f"    {bb} --> {nk}")

    # Apply styles
    for c in graphed:
        nk = _node_key(c)
        status = c.get("status", "")
        style = status_styles.get(status)
        if style:
            lines.append(f"    style {nk} {style}")

    for bb in sorted(base_builds):
        lines.append(f"    style {bb} fill:#4a148c,color:#fff")

    lines.append("```")
    lines.append("")

    return lines


def _build_rc_leaderboard(cards):
    """Build a rollout coherence leaderboard sorted best to worst."""
    scored = [c for c in cards if c.get("rollout_coherence")]
    if not scored:
        return []

    scored.sort(key=lambda c: float(c["rollout_coherence"]))

    # Find the best RC for the progress bar scaling
    rcs = [float(c["rollout_coherence"]) for c in scored]
    rc_min, rc_max = min(rcs), max(rcs)

    lines = [
        "## Rollout Coherence",
        "",
        "*Lower is better. Mean position MAE over K=20 autoregressive horizons.*",
        "",
        "| Rank | Experiment | RC | Status | Delta vs Best |",
        "|------|-----------|-----|--------|---------------|",
    ]

    best_rc = rcs[0]
    for i, c in enumerate(scored):
        cid = c.get("id", "?")
        rc = float(c["rollout_coherence"])
        status = c.get("status", "?")
        delta = rc - best_rc
        delta_str = f"+{delta:.2f}" if delta > 0 else "**best**"
        link = f"[{cid}](../run-cards/{c['filename']})"

        status_icons = {
            "kept": ":white_check_mark:",
            "discarded": ":x:",
            "running": ":hourglass:",
            "cancelled": ":stop_sign:",
        }
        icon = status_icons.get(status, "")
        lines.append(f"| {i+1} | {link} | {rc} | {icon} {status} | {delta_str} |")

    lines.append("")
    return lines


def generate_experiment_index():
    """Generate experiments/index.md from run card frontmatter."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    cards = []
    for card_path in sorted(RUN_CARDS_DIR.glob("e*.md")):
        text = card_path.read_text()
        fm = parse_yaml_frontmatter(text)
        if fm.get("id"):
            fm["filename"] = card_path.name
            cards.append(fm)

    # Sort by id
    cards.sort(key=lambda c: c.get("id", ""))

    # Group by status
    kept = [c for c in cards if c.get("status") == "kept"]
    running = [c for c in cards if c.get("status") == "running"]
    proposed = [c for c in cards if c.get("status") == "proposed"]
    discarded = [c for c in cards if c.get("status") == "discarded"]

    lines = [
        "# Experiment Index",
        "",
        f"*{len(cards)} experiments — "
        f"{len(kept)} kept, {len(running)} running, "
        f"{len(proposed)} proposed, {len(discarded)} discarded.*",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    # Best rollout coherence
    scored = [c for c in cards if c.get("rollout_coherence")]
    if scored:
        best = min(scored, key=lambda c: float(c["rollout_coherence"]))
        lines.append(
            f"**Best rollout coherence:** {best['rollout_coherence']} "
            f"([{best['id']}](../run-cards/{best['filename']}))"
        )
        lines.append("")

    # Experiment tree (Mermaid)
    lines.extend(_build_experiment_tree(cards))

    # RC leaderboard
    lines.extend(_build_rc_leaderboard(cards))

    for section_name, section_cards in [
        ("Running", running),
        ("Kept", kept),
        ("Proposed", proposed),
        ("Discarded", discarded),
    ]:
        if not section_cards:
            continue
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| ID | Type | Base | RC | Built On | Paper |")
        lines.append("|-----|------|------|----|----------|-------|")
        for c in section_cards:
            cid = c.get("id", "?")
            ctype = c.get("type", "—")
            base = c.get("base_build", "—") or "—"
            rc = c.get("rollout_coherence", "—") or "—"
            built_on = c.get("built_on", [])
            if isinstance(built_on, list):
                built_on = ", ".join(built_on) if built_on else "—"
            paper = c.get("source_paper")
            paper_str = f"[{paper}](https://arxiv.org/abs/{paper})" if paper else "—"
            link = f"[{cid}](../run-cards/{c['filename']})"
            lines.append(f"| {link} | {ctype} | {base} | {rc} | {built_on} | {paper_str} |")
        lines.append("")

    (EXPERIMENTS_DIR / "index.md").write_text("\n".join(lines))
    print(f"  generated experiments/index.md ({len(cards)} cards)")


def generate_run_cards_index():
    """Generate run-cards/index.md — listing of all run cards."""
    cards = []
    for card_path in sorted(RUN_CARDS_DIR.glob("e*.md")):
        text = card_path.read_text()
        fm = parse_yaml_frontmatter(text)
        if fm.get("id"):
            # Extract title from first heading
            title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            title = title_match.group(1) if title_match else fm["id"]
            fm["filename"] = card_path.name
            fm["title"] = title
            cards.append(fm)

    cards.sort(key=lambda c: c.get("id", ""), reverse=True)

    status_icons = {"kept": ":white_check_mark:", "running": ":hourglass:", "proposed": ":bulb:", "discarded": ":x:"}

    lines = [
        "# Run Cards",
        "",
        "Individual experiment records. Each card documents one idea tested in isolation.",
        "",
        f"*{len(cards)} cards. Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    for c in cards:
        status = c.get("status", "?")
        icon = status_icons.get(status, "")
        cid = c.get("id", "?")
        title = c.get("title", cid)
        lines.append(f"- {icon} [{title}]({c['filename']}) — *{status}*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("See [Experiment Index](../experiments/index.md) for a summary table with metrics.")

    (RUN_CARDS_DIR / "index.md").write_text("\n".join(lines))
    print(f"  generated run-cards/index.md ({len(cards)} cards)")


def generate_base_builds_index():
    """Generate base-builds/index.md from .yaml files."""
    BASE_BUILDS_DIR.mkdir(parents=True, exist_ok=True)

    builds = []
    for build_path in sorted(BASE_BUILDS_DIR.glob("b*.yaml")):
        text = build_path.read_text()
        data = parse_yaml_comments(text)
        if data.get("id"):
            data["filename"] = build_path.name
            # Extract the comment block for details
            comment_lines = [l.lstrip("# ").strip() for l in text.split("\n") if l.startswith("#")]
            data["comment_summary"] = comment_lines[0] if comment_lines else ""
            builds.append(data)

    builds.sort(key=lambda b: b.get("id", ""), reverse=True)

    lines = [
        "# Base Builds",
        "",
        "Versioned packages of canonical findings — stable sets of design decisions that experiments build on.",
        "",
        f"*{len(builds)} base build(s). Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    # Build a map of experiment id → run card filename
    card_files = {}
    all_card_paths = sorted(RUN_CARDS_DIR.glob("e*.md"))
    all_card_paths = [p for p in all_card_paths if p.name != "index.md"]
    # First pass: YAML frontmatter ids
    for card_path in all_card_paths:
        text = card_path.read_text()
        fm = parse_yaml_frontmatter(text)
        if fm.get("id"):
            card_files[fm["id"]] = card_path.name
    # Second pass: match by filename prefix (e.g., e008c → e008c-multi-position.md)
    for card_path in all_card_paths:
        stem = card_path.stem
        exp_id = stem.split("-")[0]
        if exp_id not in card_files:
            card_files[exp_id] = card_path.name

    def _link_experiment(exp_id: str) -> str:
        """Return a markdown link for an experiment id, with fallback search."""
        if exp_id in card_files:
            return f"[{exp_id}](../run-cards/{card_files[exp_id]})"
        # Try fuzzy: find any card whose filename contains the exp_id
        for card_path in all_card_paths:
            if exp_id in card_path.stem:
                return f"[{exp_id}](../run-cards/{card_path.name})"
        # Try content: find any card that mentions this exp_id
        for card_path in all_card_paths:
            if exp_id in card_path.read_text():
                return f"[{exp_id}](../run-cards/{card_path.name})"
        return exp_id

    for b in builds:
        bid = b.get("id", "?")
        desc = b.get("description", "")
        experiments = b.get("experiments", [])
        if isinstance(experiments, list):
            exp_links = [_link_experiment(exp_id) for exp_id in experiments]
            exp_str = ", ".join(exp_links)
        else:
            exp_str = str(experiments)
        created = b.get("created", "?")

        lines.append(f"## {bid}")
        lines.append("")
        lines.append(f"**{desc}**")
        lines.append("")
        lines.append(f"- **Created:** {created}")
        lines.append(f"- **Experiments:** {exp_str}")
        lines.append(f"- **Source:** [{b['filename']}]({b['filename']})")
        lines.append("")

    (BASE_BUILDS_DIR / "index.md").write_text("\n".join(lines))
    print(f"  generated base-builds/index.md ({len(builds)} builds)")


def generate_docs_index():
    """Generate docs/index.md — project landing page."""
    index_path = DOCS_DIR / "index.md"

    # Don't overwrite if it's been manually edited (has more than the template)
    if index_path.exists() and index_path.stat().st_size > 500:
        return

    lines = [
        "# Autonomous World Model",
        "",
        "A learned world model deployed onchain as an autonomous world. "
        "Trained on Melee replay data, deployed on Solana using MagicBlock's "
        "ephemeral rollups and BOLT ECS.",
        "",
        "## Quick Links",
        "",
        "- [Architecture Overview](architecture-overview.md)",
        "- [Experiment Index](experiments/index.md)",
        "- [Research Log](RESEARCH-LOG.md)",
        "- [Training Pipeline](training-pipeline-research.md)",
        "",
        "## Key Concepts",
        "",
        "- **The model IS the world.** Learned rules become ground truth.",
        "- **Arcade, not MMO.** Persistent weights/rules, not persistent world state.",
        "- **INT8 determinism for free.** Integer math is identical everywhere.",
        "",
        "---",
        "",
        "*This documentation is generated by the "
        "[simple-loop docs module](https://github.com/mattiefairchild/simple-loop) "
        "using [Zensical](https://github.com/squidfunk/zensical).*",
    ]

    index_path.write_text("\n".join(lines))
    print("  generated docs/index.md")


def copy_root_files():
    """Copy root-level files into docs/ so Zensical can serve them."""
    import shutil
    copies = [
        ("program.md", "program.md"),
    ]
    for src, dst in copies:
        src_path = PROJECT_ROOT / src
        dst_path = DOCS_DIR / dst
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  copied {src} → docs/{dst}")


def main():
    print("docs prebuild: autonomous-world-model")
    copy_root_files()
    generate_docs_index()
    generate_experiment_index()
    generate_run_cards_index()
    generate_base_builds_index()
    print("prebuild complete")


if __name__ == "__main__":
    main()
