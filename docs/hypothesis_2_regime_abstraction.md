# Hypothesis 2: Temporal Regime Abstraction in Real Capsule Endoscopy

## Hypothesis Statement

**Hypothesis 2:**  
Temporally aggregated CNN-based visual motion descriptors induce separable and temporally coherent motion regimes in real capsule endoscopy videos, enabling stable behavioral abstraction without supervision.

This hypothesis investigates whether motion behavior in real capsule endoscopy footage exhibits latent structure that can be uncovered through temporal aggregation and unsupervised clustering, without relying on pose labels or physical motion models.

---

## Experimental Setup

- **Data:** Real capsule endoscopy video (unlabelled)
- **Visual Representation:** ResNet-18 embeddings (ImageNet pretrained)
- **Motion Signal:** Frame-to-frame embedding distance
- **Temporal Aggregation:** Sliding window statistics
- **Temporal Motion Descriptor (TMD):**
  - Mean embedding change (μ)
  - Variance of embedding change (σ²)
  - Temporal Consistency Score (TCS)
- **Clustering Method:** K-means
- **Evaluated Regimes:** K = 2 and K = 3

---

## Regime Separability

Unsupervised clustering in TMD space yields distinct motion regimes.

- **K = 2:** Silhouette score = 0.4138
- **K = 3:** Silhouette score = 0.3752

The higher silhouette score for K = 2 indicates stronger separation, suggesting that capsule motion is best described by a binary abstraction rather than a higher-order partition.

---

## Temporal Stability Analysis

To evaluate whether discovered regimes correspond to stable behavioral states rather than transient noise, regime persistence and switching behavior were analyzed.

### K = 2 Regime Statistics

- Mean regime durations:
  - Regime 1: 10.62 windows
  - Regime 0: 6.81 windows
- Median duration (both regimes): 5 windows
- Flicker rate: 0.1143
- Regime dominance:
  - Regime 1: 61.1%
  - Regime 0: 38.9%

These results indicate long regime persistence, low switching frequency, and a stable dominance structure.

---

### K = 3 Regime Statistics

- Mean regime durations:
  - Regime 0: 8.65 windows
  - Regime 1: 5.50 windows
  - Regime 2: 3.37 windows
- Median durations: 3–4 windows
- Flicker rate: 0.1651
- Regime dominance:
  - Regime 0: 45.2%
  - Regime 1: 42.9%
  - Regime 2: 11.9%

The third regime exhibits shorter persistence and lower dominance, indicating that it acts as a refinement of motion behavior rather than a primary motion state.

---

## Interpretation

The results demonstrate that:

- Motion behavior in real capsule endoscopy videos exhibits **latent temporal structure**
- A **binary regime abstraction** provides the most stable and coherent representation
- Increasing the number of regimes introduces fragmentation and higher switching rates
- Regime separation functions as a **behavioral abstraction**, not a hard categorical classification

Importantly, these regimes emerge without supervision, pose estimation, or physical modeling.

---

## Conclusion

**Hypothesis 2 is supported.**  
Temporally aggregated CNN-based motion descriptors yield separable, temporally coherent motion regimes in real capsule endoscopy data. The resulting abstraction captures stable behavioral states and provides a meaningful intermediate representation for downstream motion reasoning tasks.

These findings establish regime abstraction as a principled foundation for selective motion accumulation and future pose estimation pipelines.

---

## Implications

The discovery of stable motion regimes suggests that motion estimation in capsule endoscopy should be **state-aware**. Blind accumulation of motion across all frames is likely to induce drift, while regime-aware approaches can selectively suppress unstable motion segments.

This motivates subsequent investigation into regime-aware cumulative motion estimation and its role in robust pose estimation frameworks.
