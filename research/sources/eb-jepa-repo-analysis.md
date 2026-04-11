# EB-JEPA Repo Analysis — Code-Level Reference

**Repo:** https://github.com/facebookresearch/eb_jepa
**Paper:** arXiv 2602.03604
**License:** CC-BY-NC 4.0

## Repo Structure

```
eb_jepa/
├── jepa.py           # Core JEPA class — unroll(), plan(), loss computation
├── planning.py       # CEM and MPPI planners
├── models/
│   ├── encoders.py   # ResNet5, ImpalaEncoder, ResNet18
│   ├── predictors.py # ResUNet, RNNPredictor (GRU)
│   └── losses.py     # HingeStdLoss, CovarianceLoss, TemporalSimilarityLoss, InverseDynamicsLoss
├── examples/
│   ├── image_jepa/   # CIFAR-10 image masking
│   ├── video_jepa/   # Moving MNIST video prediction
│   └── ac_video_jepa/# Action-Conditioned — Two Rooms world model + planning
└── tests/
```

## Key Architecture: Action-Conditioned Video JEPA

### Encoder: ImpalaEncoder
- Lightweight CNN for the Two Rooms environment
- Maps observation → latent state vector

### Predictor: RNNPredictor (GRU)
- Single-step state propagation: `(state, action) → next_state`
- GRU cell conditioned on actions
- This is the dynamics model

### Anti-Collapse: VC Regularization
```python
# From losses.py
class HingeStdLoss:
    """Penalizes per-feature std below margin — prevents dimensional collapse"""
    
class CovarianceLoss:
    """Penalizes off-diagonal covariance — prevents feature redundancy"""
    
class TemporalSimilarityLoss:
    """Encourages smooth consecutive representations"""
    
class InverseDynamicsLoss:
    """Predicts actions from consecutive state pairs — grounds in dynamics"""
```

### Unrolling (jepa.py)
```python
def unroll(self, obs, actions, mode='parallel'):
    if mode == 'parallel':
        # All timesteps at once, ground truth re-fed (training)
        ...
    elif mode == 'autoregressive':
        # Step-by-step, sliding window, uses own predictions (inference)
        for t in range(horizon):
            context = states[-N:]  # last N states
            next_state = self.predictor(context, action[t])
            states.append(next_state)
```

### Planning (planning.py)
```python
class CEMPlanner:
    """Cross-Entropy Method for action optimization in latent space"""
    def plan(self, current_obs, goal_obs, jepa_model):
        current_emb = jepa_model.encode(current_obs)
        goal_emb = jepa_model.encode(goal_obs)
        
        for iteration in range(n_iterations):
            # Sample action sequences from current distribution
            actions = sample(mean, std, n_samples)
            
            # Rollout each sequence through world model
            final_embs = jepa_model.unroll(current_emb, actions, mode='autoregressive')
            
            # Score by distance to goal
            costs = mse(final_embs, goal_emb)
            
            # Update distribution from elite samples
            elite_idx = topk(costs, n_elite)
            mean, std = fit(actions[elite_idx])
        
        return mean[0]  # first action of best sequence

class MPPIPlanner:
    """Model Predictive Path Integral — soft reweighting instead of hard elite selection"""
```

## Key Differences from LeWM

| Aspect | EB-JEPA | LeWM |
|--------|---------|------|
| Anti-collapse | VC regularization (4 loss terms) | SIGReg (1 term, 1 hyperparameter) |
| Encoder | Lightweight CNN (ImpalaEncoder) | ViT-tiny |
| Predictor | GRU (recurrent) | Transformer (attention) |
| Target encoder | Same encoder (no EMA) | Same encoder (no EMA) |
| Purpose | Pedagogical / modular | Research result |
| Environments | Two Rooms only | 4 environments |

## Relevance to AWM Adaptation

The EB-JEPA AC Video JEPA example is the clearest template for what a "JEPA world model" looks like at minimum:
1. Encoder maps observations → latent
2. GRU predictor: (latent_state, action) → next_latent_state
3. Autoregressive rollout for multi-step prediction
4. Planning via CEM in latent space

For Melee, the adaptation would replace:
- Pixel encoder → game state encoder (MLP or learned embeddings on our structured features)
- GRU → could keep GRU, or use Transformer/Mamba predictor
- CEM planning → not needed for world model evaluation, but interesting for "The Wire"
