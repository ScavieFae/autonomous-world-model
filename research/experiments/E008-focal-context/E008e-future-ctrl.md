# E008e: Future Controller Conditioning

**Variant of:** E008 (focal context)
**Category:** data conditioning
**Complexity:** ~30 lines (encoding + dataset)
**Status:** complete
**Branch:** scav/E008e-future-ctrl
**wandb:** `ok5bww53`

## Idea

Mattie's idea: instead of future *state* (which you don't have at inference), give the model future *controller inputs* (which you could have via speculative lookahead or opponent modeling). Predict frame t, but condition on ctrl at T, T+1, T+2.

Motivation: controller inputs are available before the state they produce (via rollback-style prediction or GGPO input delay). If the model can use future ctrl, it would know "player is about to press B" before seeing the resulting state.

## Results (2K games, 2 epochs, H100)

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| change_acc | 64.7% | -2.6pp |
| action_acc | 96.4% | — |
| pos_mae | 0.65 | same |
| val_loss | 0.307 | +0.016 (worse) |

**Verdict:** Below baseline. The extra ctrl signal didn't help — the model may need more epochs to learn to use it, or future button presses alone (without future state) aren't informative enough. At 60fps, ctrl at t+1 and t+2 is ~80% identical to ctrl at t (human motor autocorrelation). The 5% of frames with discrete transitions are where future ctrl differs, but without knowing the future actionability state, the model can't tell if the input will take effect. The extra conditioning dimensions (78 vs 26) may have added noise that hurt.

See also: `research/notes/rollback-delay-frames.md` for the GGPO connection.
