"""
Diagnose hitbox table coverage gaps against training data.

Loads hitbox_table.json and scans 20 parquet files from parsed training data
to find which attack frames are missing from the hitbox lookup table.
"""

import json
import os
import zlib
import io
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq


# Melee character names by internal ID
CHAR_NAMES = {
    0: "Mario", 1: "Fox", 2: "Captain Falcon", 3: "DK", 4: "Kirby",
    5: "Bowser", 6: "Link", 7: "Sheik", 8: "Ness", 9: "Peach",
    10: "Ice Climbers", 11: "Nana", 12: "Pikachu", 13: "Samus", 14: "Yoshi",
    15: "Jigglypuff", 16: "Mewtwo", 17: "Luigi", 18: "Marth", 19: "Zelda",
    20: "Young Link", 21: "Dr. Mario", 22: "Falco", 23: "Pichu",
    24: "Game & Watch", 25: "Ganondorf", 26: "Roy",
}

# Attack action state categories
GROUND_NORMALS = set(range(44, 65))      # 44-64: jab, tilts, smashes, dash attack
AERIALS = set(range(65, 75))             # 65-74: nair, fair, bair, uair, dair + landing lags
GRABS_THROWS = set(range(212, 228))      # 212-227: grabs and throws
SPECIALS_START = 341                      # 341+: character-specific specials

# Also include some commonly seen attack-adjacent actions
# 70-74 are landing lag from aerials, still attack states
# 233-238 are additional grab/throw states for some chars
EXTRA_GRABS = set(range(228, 241))

def is_attack_action(action_id: int) -> bool:
    """Check if an action state ID corresponds to an attack."""
    return (
        action_id in GROUND_NORMALS
        or action_id in AERIALS
        or action_id in GRABS_THROWS
        or action_id in EXTRA_GRABS
        or action_id >= SPECIALS_START
    )


def categorize_action(action_id: int) -> str:
    """Categorize an action ID into a human-readable group."""
    if action_id in GROUND_NORMALS:
        return "ground_normals (44-64)"
    elif action_id in AERIALS:
        return "aerials (65-74)"
    elif action_id in GRABS_THROWS:
        return "grabs_throws (212-227)"
    elif action_id in EXTRA_GRABS:
        return "extra_grabs (228-240)"
    elif action_id >= SPECIALS_START:
        return f"specials (341+)"
    else:
        return f"other ({action_id})"


def load_hitbox_table(path: str) -> dict:
    """Load hitbox table and return the inner 'table' dict."""
    with open(path) as f:
        data = json.load(f)
    return data["table"]


def load_parquet_file(path: str) -> pd.DataFrame:
    """Load a zlib-compressed parquet file and flatten to columns we need."""
    with open(path, "rb") as f:
        raw = f.read()
    decompressed = zlib.decompress(raw)
    table = pq.read_table(io.BytesIO(decompressed))

    # Extract nested struct fields we need
    root = table.column("root")

    # We need p0 and p1 action, state_age, character
    rows = []
    for i in range(table.num_rows):
        row = root[i].as_py()
        p0 = row["p0"]
        p1 = row["p1"]
        rows.append({
            "p0_action": p0["action"],
            "p0_state_age": p0["state_age"],
            "p0_character": p0["character"],
            "p1_action": p1["action"],
            "p1_state_age": p1["state_age"],
            "p1_character": p1["character"],
        })

    return pd.DataFrame(rows)


def main():
    hitbox_path = "/Users/mattiefairchild/claude-projects/nojohns/worldmodel/research/hitbox-data/hitbox_table.json"
    data_dir = os.path.expanduser("~/claude-projects/nojohns-training/data/parsed-v2/games/")

    print("Loading hitbox table...")
    hitbox_table = load_hitbox_table(hitbox_path)

    # Pre-compute what's in the table
    table_keys = set(hitbox_table.keys())
    # Also index by (char, action) to distinguish "action not in table" vs "frame too high"
    table_char_actions = defaultdict(set)  # (char, action) -> set of frames
    for key in table_keys:
        parts = key.split(":")
        char_id, action_id, frame = int(parts[0]), int(parts[1]), int(parts[2])
        table_char_actions[(char_id, action_id)].add(frame)

    print(f"Hitbox table: {len(table_keys)} entries, {len(table_char_actions)} (char, action) pairs")

    # Scan parquet files
    files = sorted(os.listdir(data_dir))[:20]
    print(f"\nScanning {len(files)} parquet files...")

    # Counters
    total_attack_frames = 0
    total_hit_frames = 0
    total_miss_frames = 0

    # Miss categorization
    miss_by_reason = Counter()       # "char_missing", "action_missing", "frame_too_high"
    miss_by_char = Counter()         # char_id -> count
    miss_by_action = Counter()       # action_id -> count
    miss_by_char_action = Counter()  # (char_id, action_id) -> count
    miss_by_category = Counter()     # category string -> count
    hit_by_category = Counter()      # category string -> count

    # Also track all attack actions seen (for context)
    all_attack_actions_seen = Counter()  # action_id -> count
    all_attack_chars_seen = Counter()    # char_id -> count

    for fi, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        try:
            df = load_parquet_file(path)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        if fi % 5 == 0:
            print(f"  Processing file {fi+1}/{len(files)}...")

        # Check both players
        for prefix in ["p0", "p1"]:
            actions = df[f"{prefix}_action"].values
            state_ages = df[f"{prefix}_state_age"].values
            characters = df[f"{prefix}_character"].values

            for action, state_age, char in zip(actions, state_ages, characters):
                if not is_attack_action(action):
                    continue

                frame = int(state_age) + 1  # state_age is 0-indexed, hitbox table is 1-indexed
                total_attack_frames += 1
                category = categorize_action(action)
                all_attack_actions_seen[action] += 1
                all_attack_chars_seen[char] += 1

                key = f"{char}:{action}:{frame}"

                if key in table_keys:
                    total_hit_frames += 1
                    hit_by_category[category] += 1
                else:
                    total_miss_frames += 1
                    miss_by_char_action[(char, action)] += 1
                    miss_by_char[char] += 1
                    miss_by_action[action] += 1
                    miss_by_category[category] += 1

                    # Classify the reason
                    chars_in_table = {ca[0] for ca in table_char_actions}
                    if char not in chars_in_table:
                        miss_by_reason["char_not_in_table"] += 1
                    elif (char, action) not in table_char_actions:
                        miss_by_reason["action_not_in_table_for_char"] += 1
                    else:
                        max_frame = max(table_char_actions[(char, action)])
                        if frame > max_frame:
                            miss_by_reason["frame_exceeds_table_max"] += 1
                        elif frame < min(table_char_actions[(char, action)]):
                            miss_by_reason["frame_below_table_min"] += 1
                        else:
                            miss_by_reason["frame_gap_in_table"] += 1

    # ── Results ──
    print("\n" + "=" * 80)
    print("HITBOX TABLE COVERAGE DIAGNOSTIC")
    print("=" * 80)

    coverage = total_hit_frames / total_attack_frames * 100 if total_attack_frames > 0 else 0
    print(f"\nOverall: {total_hit_frames}/{total_attack_frames} attack frames covered ({coverage:.1f}%)")
    print(f"  Hits:   {total_hit_frames}")
    print(f"  Misses: {total_miss_frames}")

    # ── Coverage by category ──
    print(f"\n{'─' * 60}")
    print("COVERAGE BY ATTACK CATEGORY")
    print(f"{'─' * 60}")
    all_categories = sorted(set(list(hit_by_category.keys()) + list(miss_by_category.keys())))
    for cat in all_categories:
        hits = hit_by_category[cat]
        misses = miss_by_category[cat]
        total = hits + misses
        pct = hits / total * 100 if total > 0 else 0
        print(f"  {cat:30s}  {hits:>7d}/{total:>7d}  ({pct:5.1f}%)")

    # ── Miss reasons ──
    print(f"\n{'─' * 60}")
    print("MISS REASONS")
    print(f"{'─' * 60}")
    for reason, count in miss_by_reason.most_common():
        pct = count / total_miss_frames * 100
        print(f"  {reason:40s}  {count:>7d}  ({pct:5.1f}%)")

    # ── Top 20 missing (char, action) pairs ──
    print(f"\n{'─' * 60}")
    print("TOP 30 MISSING (character, action) PAIRS")
    print(f"{'─' * 60}")
    print(f"  {'Character':20s}  {'CharID':>6s}  {'ActID':>6s}  {'Count':>7s}  {'Category':30s}  {'Reason'}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*30}  {'─'*30}")
    for (char, action), count in miss_by_char_action.most_common(30):
        char_name = CHAR_NAMES.get(char, f"Unknown({char})")
        category = categorize_action(action)

        chars_in_table = {ca[0] for ca in table_char_actions}
        if char not in chars_in_table:
            reason = "CHAR NOT IN TABLE"
        elif (char, action) not in table_char_actions:
            reason = "ACTION NOT IN TABLE"
        else:
            max_f = max(table_char_actions[(char, action)])
            min_f = min(table_char_actions[(char, action)])
            reason = f"frames {min_f}-{max_f} in table"

        print(f"  {char_name:20s}  {char:>6d}  {action:>6d}  {count:>7d}  {category:30s}  {reason}")

    # ── Characters with worst coverage ──
    print(f"\n{'─' * 60}")
    print("COVERAGE BY CHARACTER")
    print(f"{'─' * 60}")
    all_chars = sorted(set(list(all_attack_chars_seen.keys())))
    char_results = []
    for char in all_chars:
        total = all_attack_chars_seen[char]
        misses = miss_by_char.get(char, 0)
        hits = total - misses
        pct = hits / total * 100 if total > 0 else 0
        char_results.append((char, total, hits, misses, pct))

    # Sort by coverage % ascending (worst first)
    char_results.sort(key=lambda x: x[4])
    print(f"  {'Character':20s}  {'CharID':>6s}  {'Total':>7s}  {'Hits':>7s}  {'Misses':>7s}  {'Coverage':>8s}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    for char, total, hits, misses, pct in char_results:
        char_name = CHAR_NAMES.get(char, f"Unknown({char})")
        print(f"  {char_name:20s}  {char:>6d}  {total:>7d}  {hits:>7d}  {misses:>7d}  {pct:>7.1f}%")

    # ── Most common attack actions seen that are NOT in table at all ──
    print(f"\n{'─' * 60}")
    print("ACTION IDs IN TRAINING DATA NOT IN HITBOX TABLE (any character)")
    print(f"{'─' * 60}")
    # Collect all action IDs that appear in misses with reason "action not in table"
    actions_not_in_table = set()
    for (char, action), count in miss_by_char_action.items():
        if (char, action) not in table_char_actions:
            actions_not_in_table.add(action)

    # But some actions may exist for SOME characters. Show which.
    actions_in_table = {ca[1] for ca in table_char_actions}
    fully_missing = actions_not_in_table - actions_in_table
    partially_missing = actions_not_in_table & actions_in_table

    if fully_missing:
        print("\n  Completely absent from table (no character has them):")
        for action in sorted(fully_missing):
            count = miss_by_action[action]
            print(f"    Action {action:>4d}: {count:>7d} frames  ({categorize_action(action)})")

    if partially_missing:
        print("\n  Present for some characters but not others:")
        for action in sorted(partially_missing):
            # Which chars have it vs which need it
            chars_with = {ca[0] for ca in table_char_actions if ca[1] == action}
            chars_need = {ca[0] for ca, c in miss_by_char_action.items() if ca[1] == action and (ca[0], action) not in table_char_actions}
            count = sum(c for (ch, ac), c in miss_by_char_action.items() if ac == action and (ch, action) not in table_char_actions)
            chars_with_names = [CHAR_NAMES.get(c, str(c)) for c in sorted(chars_with)]
            chars_need_names = [CHAR_NAMES.get(c, str(c)) for c in sorted(chars_need)]
            print(f"    Action {action:>4d}: {count:>7d} miss frames")
            print(f"      Have: {', '.join(chars_with_names)}")
            print(f"      Need: {', '.join(chars_need_names)}")

    # ── Frame overflow analysis ──
    print(f"\n{'─' * 60}")
    print("FRAME OVERFLOW ANALYSIS (action in table but state_age exceeds max)")
    print(f"{'─' * 60}")
    overflow_pairs = []
    for (char, action), count in miss_by_char_action.items():
        if (char, action) in table_char_actions:
            max_f = max(table_char_actions[(char, action)])
            overflow_pairs.append((char, action, count, max_f))
    overflow_pairs.sort(key=lambda x: -x[2])
    if overflow_pairs:
        print(f"  {'Character':20s}  {'ActID':>6s}  {'Misses':>7s}  {'MaxFrame':>8s}")
        print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*8}")
        for char, action, count, max_f in overflow_pairs[:20]:
            char_name = CHAR_NAMES.get(char, f"Unknown({char})")
            print(f"  {char_name:20s}  {action:>6d}  {count:>7d}  {max_f:>8d}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
