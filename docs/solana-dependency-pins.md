# Solana Dependency Pins — Operating Guide

This doc explains why certain dependencies are pinned in `solana/Cargo.lock` and how to handle version conflicts when upgrading.

## The Core Problem

The SBF (Solana BPF) toolchain ships its own Cargo, which lags behind the host Cargo. As of platform-tools v1.51, SBF Cargo is **1.84**. Rust `edition = "2024"` requires Cargo **1.85+**. Any transitive dependency that ships with `edition = "2024"` in its Cargo.toml will fail at download time — before resolution even starts.

This means the SBF build can break without any code changes, just because a transitive dep published a new minor version.

## Current Pins

| Crate | Pinned To | Why | Remove When |
|-------|-----------|-----|-------------|
| `blake3` | 1.7.0 | 1.8+ uses `edition = "2024"` | platform-tools ships Cargo 1.85+ |
| `constant_time_eq` | 0.3.1 | 0.4+ uses `edition = "2024"` (dep of blake3) | same as above |
| `session-keys` | 2.0.7 | Only version compatible with both anchor-lang 0.31.1 and bolt-lang 0.2.4's zeroize requirement (see below) | bolt-lang upgrades past 0.2.4 |

## The session-keys Version Squeeze

`bolt-lang 0.2.4` hard-depends on `session-keys ^2`. The available versions form a trap:

| session-keys | anchor-lang req | Problem |
|---|---|---|
| 2.0.8 | 0.32.1 | `native_token` moved to own crate in solana-program 4.0 — compile error |
| 2.0.6 | =0.30.1 | Exact pin, incompatible with our 0.31.1 |
| 2.0.5 | =0.30.1 | Same |
| 2.0.4 | =0.30.0 | Same |
| 2.0.3 | ^0.29.0 | Pulls solana-program 1.x → zeroize <1.4 conflicts with bolt-lang's ^1.7 |
| 2.0.2 | ^0.28.0 | Same |
| **2.0.7** | **>=0.30.0** | Works with 0.31.1, no zeroize conflict |

**Only 2.0.7 works.** This is fragile. If session-keys publishes 2.0.9 with another anchor-lang bump, it may get auto-selected and break.

## How to Re-Pin After Lockfile Changes

If someone deletes `Cargo.lock` or runs a bare `cargo update`:

```bash
cd solana

# Generate fresh lockfile with host cargo (can parse edition 2024)
cargo generate-lockfile

# Pin problematic crates
cargo update --package blake3 --precise 1.7.0
cargo update --package session-keys --precise 2.0.7

# Verify
anchor build
```

## How to Diagnose New Breakages

### Symptom: `feature 'edition2024' is required`

A transitive dep published a new version using edition 2024. Find it:

```bash
# This runs with host cargo, which can resolve
cargo tree -i <crate-name>
```

Then pin the offending crate to its last edition-2021 version:

```bash
cargo update --package <crate> --precise <version>
```

### Symptom: `failed to select a version for X`

Version conflict in the resolution graph. Debug with:

```bash
cargo tree -i <conflicting-crate>   # who needs it
cargo tree -e features               # what features are requested
```

Check crates.io for the dependency requirements of each version to find one that satisfies all constraints.

### Symptom: `unresolved import` or `can't find crate`

The BOLT macros (`#[component_deserialize]`, `#[system]`, `#[arguments]`) expand to code that uses certain crates/traits:

- `#[component_deserialize]` auto-derives `Clone`, `Copy`, `AnchorSerialize`, `AnchorDeserialize` — don't derive them manually
- `#[arguments]` needs `serde` with `derive` feature in Cargo.toml
- `#[system_input]` references component types by name — ensure the corresponding `use` import exists

## When Upgrading bolt-lang

When upgrading past 0.2.4:

1. Check what `session-keys` version the new bolt-lang requires
2. Check what `anchor-lang` version that session-keys needs
3. Check whether any transitive deps use edition 2024
4. Run the pin sequence above if needed
5. Run `anchor build` and fix any macro expansion changes

## Workspace Cargo.toml

The workspace has `blake3 = "=1.7.0"` and `constant_time_eq = "=0.3.1"` in `[workspace.dependencies]`. These serve as documentation — the actual enforcement is the lockfile. If a member crate doesn't reference them with `workspace = true`, they don't affect resolution. The lockfile pins are what matter.
