# Research Canon

The key papers and projects in world model research. Curated by [General Intuition](https://generalintuition.com/) and [Packy McCormick (Not Boring)](https://www.notboring.co/p/world-models).

---

## Paradigms

- **Making the World Differentiable** (Schmidhuber, 1990) — First proposal for a neural network that learns a predictive model of the environment for planning.
- **Dyna** (Sutton, 1991) — Model-based RL framework: learn a world model, use it to generate simulated experience for planning.
- **World Models** (Ha & Schmidhuber, 2018) — VAE + RNN world model trained on environment frames; agent learns policy entirely inside the dream. The paper that named the field.
- **SimPLe** (Kaiser et al., 2019) — Simulated Policy Learning. Video prediction model as a learned simulator for Atari, enabling 100x sample efficiency.
- **PlaNet** (Hafner et al. / DeepMind, 2019) — Learning latent dynamics models for planning from pixels. Pure model-based control without a policy network.
- **DreamerV1** (Hafner et al. / Google Brain, 2019) — Dream to Control. Learns behaviors by backpropagating through learned world model dynamics.
- **DreamerV2** (Hafner / Google Brain, 2021) — Discrete latent representations for world models. First to achieve human-level on Atari from pixels with a world model.
- **DreamerV3** (DeepMind, 2023) — General-purpose world model agent. Single fixed hyperparameter set works across diverse domains — Atari, DMC, Minecraft.
- **NeRF** (Mildenhall et al., 2020) — Neural Radiance Fields. Learns a continuous 3D scene representation from 2D images. Not RL, but foundational for spatial world models.
- **IRIS** (Micheli, Alonso, Fleuret, 2022) — Transformer world model operating on discrete tokens (VQ-VAE). Sample-efficient Atari agent.
- **Delta-IRIS** (Micheli, Alonso, Fleuret, 2024) — Predicts frame deltas instead of full frames. More efficient, better at long-horizon coherence.
- **A Path Towards Autonomous Machine Intelligence** (LeCun, 2022) — Position paper proposing JEPA as an alternative to generative models. Argues world models should predict in latent space, not pixel space.
- **JEPA** (LeCun / Meta, 2023) — Joint Embedding Predictive Architecture. Predicts latent representations of targets rather than reconstructing inputs directly.
- **MC-JEPA** (LeCun / Meta, 2023) — Multi-modal Contrastive JEPA. Extends joint embedding prediction across vision and other modalities.
- **V-JEPA 2** (LeCun / Meta, 2025) — Video JEPA at scale. Self-supervised video understanding without pixel-level reconstruction.
- **LeJEPA** (Balestriero & LeCun / Meta FAIR, 2025) — Latent Energy JEPA. Energy-based formulation of the joint embedding predictive architecture.
- **GAIA-1** (Wayve, 2023) — Generative world model for autonomous driving. Learns road dynamics from video, generates realistic driving scenarios.
- **DIAMOND** (Alonso, Jelley, Micheli et al., 2024) — Diffusion as world model. Uses diffusion process to generate next frames in Atari. State-of-the-art sample efficiency.
- **GAIA-2** (Wayve, 2025) — Next-generation driving world model. Improved fidelity and controllability over GAIA-1.
- **Genie** (Google DeepMind, 2024) — Generative Interactive Environments. Learns playable 2D worlds from unlabeled video. Infers latent actions without action labels.
- **SORA 2** (OpenAI, 2025) — Video generation model framed as world simulation. Generates temporally coherent video with implicit physics.
- **Veo 3** (Google DeepMind, 2025) — High-fidelity video generation with audio. Extends world simulation to multimodal output.
- **Learning to Drive from a World Model** (Comma.ai, 2025) — End-to-end driving from a learned world model deployed in production vehicles.
- **SIMA 2** (Google DeepMind, 2025) — Scalable Instructable Multiworld Agent. Follows language instructions across multiple 3D game environments.
- **Navigation World Models** (FAIR at Meta, Berkeley, NYU, 2025) — World models for embodied navigation. Predicts visual observations from actions in real environments.
- **VIPER** (Google DeepMind, 2023) — Video prediction for environment rollouts. Uses video prediction models as learned simulators for RL.
- **Self-Forcing** (Adobe, UT Austin, 2025) — Trains autoregressive models by forcing them to consume their own predictions during training, closing the train/inference gap.
- **Learning to Model the World with Language** (DeepMind, 2024) — Dynalang. Grounds language in world model dynamics — the model predicts future states conditioned on language.

## Infrastructure

- **SimpleFSDP** (Meta, UCSD, 2024) — Simplified Fully Sharded Data Parallel. Cleaner FSDP implementation for distributed training.
- **Meta Lingua** (Meta, 2024) — Minimal, research-friendly language model training framework. Clean reference implementation.
- **Mamba in the Llama** (Meta, 2024) — Hybrid Mamba-transformer architectures. Distills transformer knowledge into SSM layers for efficient inference.
- **SigLIP 2** (Google DeepMind, 2025) — Scalable vision-language encoder. Improved contrastive learning for image-text alignment.
- **DINOv3** (Meta AI, 2025) — Self-supervised vision foundation model. Learns visual representations without labels at scale.

---

*Source: [The World Model Research Archive](https://www.notboring.co/p/world-models) by General Intuition x Not Boring (Packy McCormick)*
