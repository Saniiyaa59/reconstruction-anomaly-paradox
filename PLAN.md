# Project Plan

CS599 Deep Visual Generative Models, final project (blog). Deadline: mid-December 2026.

## Goals

**Main question:** reconstruction-based anomaly detectors fail when they get too good at
reconstructing. Do the parts that make a model *generative* (the VAE's KL prior, diffusion's
noise schedule, scoring by likelihood) protect against that failure, or make it worse?

Specifically:
1. **Capacity paradox:** does detection AUROC collapse as capacity grows in a VAE and a
   diffusion model, as it does for PCA? What is each model's capacity knob?
2. **Reconstruction vs likelihood:** do reconstruction error and likelihood (ELBO) agree on what
   counts as anomalous? When they disagree, why? (Split the ELBO into its reconstruction and KL terms.)
3. **Two failure modes:** separate failure caused by *too much capacity* (one-class, same-domain
   anomalies) from failure caused by *anomalies that are simpler than the training data*
   (cross-dataset, Nalisnick et al.).

## Experiments

Every experiment trains on normal data only, sweeps a capacity knob, and reports at each setting:
mean normal and anomaly error, overall AUROC, and AUROC per anomaly class. Anomaly scores:
reconstruction error and likelihood (−ELBO / diffusion likelihood bound); for the VAE also the KL
term alone, as a diagnostic. 3 seeds per setting.

| # | Tier | Setup | Normal | Anomalies | Purpose |
|---|---|---|---|---|---|
| E1 | core | ECG5000 | class 1 | classes 2–5 | 1D result; matches the internship story |
| E2 | core | One-class MNIST | digit 8 (also 0 if time) | other digits | capacity paradox on images; digit 1 shows the simplicity failure |
| E3 | stretch 1 | FashionMNIST → MNIST | FashionMNIST | MNIST | failure due to simpler anomalies (Nalisnick) |
| E4 | stretch 1 | MNIST → FashionMNIST | MNIST | FashionMNIST | control; this direction is expected to work |
| E5 | stretch 2 | MVTec AD or CIFAR-10 → SVHN | defect-free / CIFAR-10 | defects / SVHN | only if far ahead |

All experiments, core and stretch, use the full capacity sweep.

**Training losses:** VAE = negative ELBO (reconstruction + β·KL). Diffusion = standard
noise-prediction MSE. Reconstruction error is the *score* used at test time, not the training loss.

**Capacity knobs:**
- PCA: number of components k (done).
- VAE: latent dimension z ∈ {1…128} at β = 1; β ∈ {0.01, 0.1, 1, 4} at large z; optionally input
  resolution. Log active latent units at each setting.
- Diffusion: test-time noise level t/T ∈ {0.05…0.9} (no retraining needed); optionally model width.

**Hypotheses:**
- E1/E2: AUROC is high at low capacity and collapses at high capacity. The VAE's KL term may slow
  the collapse unless β is small. Diffusion gives an inverted U over t.
- E3: AUROC is poor at *every* capacity, because MNIST is simpler than FashionMNIST.
- E4: AUROC is good. If so, E3's failure is caused by input simplicity, not a bug.

## Status

- [x] ECG5000 download + splits (`src/data.py`): 1751 train / 291 val / 877 normal + 2081 anomalous test
- [x] Shared evaluation + paradox-curve plot (`src/evaluate.py`)
- [x] PCA sweep on ECG (`src/pca_model.py`): AUROC 0.99 at k=2 → 0.28 at k=120
- [x] PCA sweep on one-class MNIST (`--dataset mnist --normal {0,1,8}`): **no collapse**
      (normal 0: 0.98 → min 0.92; normal 1: ~0.99 throughout; normal 8: 0.85 → 0.77). Pixels that never
      contain ink in the training digit can't be reconstructed by PCA at any k. With normal = 8,
      digit 1 scores AUROC ~0.2–0.35 at every k (simplicity failure).
- [ ] Conv autoencoder check on one-class MNIST (does AUROC fall with width/latent size?).
      If it stays flat like PCA, switch E2 to FashionMNIST classes or MVTec.
- [ ] VAE on ECG (E1)
- [ ] Diffusion on ECG (E1)
- [ ] Image data loaders + 2D conv networks
- [ ] VAE + diffusion on E2–E4
- [ ] Blog

## Timeline

| Dates | Work |
|---|---|
| Oct 3–17 | VAE on ECG: z and β sweeps, both scores |
| Oct 17–31 | Diffusion on ECG: t sweep, both scores |
| Nov 1–21 | E2 with VAE + diffusion; then E3/E4 if on track |
| Nov 21–30 | Buffer (Thanksgiving); E5 only if well ahead |
| Dec 1–mid | Blog writing + final figures |

## Code design

- The network is swappable per dataset: MLP / 1D conv for ECG, 2D conv for images. The training
  loop, sweep script and `evaluate.py` are shared.
- `evaluate.py` takes flat per-sample scores and class labels, so it already works with any dataset.
- Results go to `results/metrics/*.csv` and `results/figures/`.

## Blog outline

1. Hook: the internship autoencoder that got worse as resolution increased.
2. How reconstruction-based anomaly detection works.
3. PCA sanity check: the paradox in its simplest form.
4. VAE and diffusion on ECG: does being generative help?
5. Reconstruction vs likelihood: two scores that measure different things.
6. Images: capacity failure (one-class) vs simplicity failure (cross-dataset).
7. What this means for anomaly detectors in real deployments.

## Key references

- Nalisnick et al. (2019), "Do Deep Generative Models Know What They Don't Know?"
- Serrà et al. (2020), "Input Complexity and Out-of-Distribution Detection with Likelihood-based Generative Models"
- Theis et al. (2016), "A note on the evaluation of generative models"
- Ruff et al. (2018), "Deep One-Class Classification" (one-class MNIST protocol)
- Diffusion-based anomaly detection (e.g. DiffusionAD), on how detection depends on noise strength
