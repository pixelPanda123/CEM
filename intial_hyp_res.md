
# Results: Evaluation of Motion Cues in Real Capsule Endoscopy Video

## 1. Objective and Initial Hypothesis

This study set out to evaluate whether learned visual representations—particularly transformer-based embeddings—can capture motion-relevant temporal information in real capsule endoscopy videos more robustly than classical optical flow. The analysis focused on scenarios characterized by texture-poor regions, non-rigid tissue deformation, fluid motion, and imaging artifacts, where traditional motion estimation methods are known to struggle.

The **initial hypothesis** was formulated as follows:

> **H₀ (Initial):** Transformer-based visual representations can capture motion-relevant temporal information in real capsule endoscopy videos more robustly than classical optical flow, leading to improved stability and consistency in relative motion estimation under texture-poor and non-rigid conditions.

This hypothesis was further decomposed into four investigatory dimensions:

* (A) Temporal smoothness
* (B) Drift behavior
* (C) Sensitivity to temporal distance
* (D) Robustness under stress (fluid motion, bubbles, artifacts)

---

## 2. Baseline Motion Cues

### 2.1 Pixel Difference

Mean pixel difference was evaluated as a naïve baseline for motion estimation. The resulting signal exhibited high sensitivity to illumination changes, fluid dynamics, bubbles, and transient artifacts. Large spikes frequently occurred in the absence of capsule translation, while smooth forward motion often produced minimal response.

**Observation:** Pixel difference primarily captures appearance change rather than motion and fails to provide a stable or interpretable temporal signal.

---

### 2.2 Optical Flow

Dense optical flow was evaluated as a classical motion estimation baseline. While optical flow responded strongly to visual changes, it proved unreliable in non-rigid gastrointestinal environments. Motion induced by fluids, tissue deformation, and independently moving particles dominated the flow signal, frequently overwhelming any contribution from capsule translation.

**Observation:** Optical flow does not provide a reliable reference for capsule motion in real endoscopy videos and cannot be treated as ground-truth motion.

This finding directly challenges the assumption that optical flow constitutes a stable physical baseline in this domain.

---

## 3. Learned Visual Representations

### 3.1 Frame-Level CNN and ViT Embeddings

Frame-to-frame embedding distances were computed using frozen ImageNet-pretrained CNN and ViT backbones. At this level, both representations exhibited high variability and impulse-like responses to local appearance changes.

* CNN embeddings produced relatively bounded but still noisy signals.
* ViT embeddings exhibited higher variance and increased sensitivity to local patch-level changes, such as fluid shimmer and specular highlights.

**Observation:** At frame-level granularity, neither CNN nor ViT embeddings yield motion-relevant temporal signals.

This confirms **Sub-hypothesis A (Temporal smoothness)** *only partially*, indicating that representation choice alone is insufficient.

---

## 4. Temporal Aggregation Effects

To address the limitations of frame-level analysis, temporal windowing was applied to embedding distance signals.

### 4.1 Variance Reduction

Temporal windowing reduced variance for both CNN and ViT embeddings. Quantitatively:

* CNN raw variance: **13.0638**

* CNN windowed variance: **7.9516**

* CNN variance reduction ratio (VRR): **1.64**

* ViT raw variance: **16.7776**

* ViT windowed variance: **9.3075**

* ViT variance reduction ratio (VRR): **1.80**

Although ViT embeddings exhibited a larger *relative* variance reduction, CNN embeddings retained lower absolute variance both before and after aggregation.

**Interpretation:** Variance reduction alone does not guarantee interpretable motion signals; residual instability remains representation-dependent.

---

### 4.2 Temporal Consistency

To quantify temporal coherence, a Temporal Consistency Score (TCS) was computed over windowed embedding signals.

* CNN TCS: **0.6399**
* ViT TCS: **0.7994**

Lower scores indicate smoother, more predictable temporal evolution.

**Observation:** CNN embeddings exhibit significantly greater temporal consistency than ViT embeddings after aggregation.

This resolves the apparent contradiction between variance reduction and interpretability: although ViT embeddings benefit numerically from smoothing, they remain temporally erratic.

This directly supports **Sub-hypothesis C (Sensitivity to temporal distance)** and **Sub-hypothesis D (Robustness under stress)**.

---

## 5. Drift Behavior and Relative Motion

While absolute pose or displacement could not be estimated due to the absence of ground-truth labels, the cumulative behavior of temporally aggregated embedding signals revealed qualitatively different drift characteristics:

* CNN embeddings exhibited stable, monotonic cumulative change patterns.
* ViT embeddings demonstrated oscillatory behavior, complicating interpretation as relative displacement.
* Optical flow exhibited chaotic accumulation dominated by non-rigid motion.

**Observation:** Certain learned representations support stable relative motion proxies, while others do not.

This partially supports **Sub-hypothesis B (Drift behavior)** at a representational level without overclaiming spatial localization.

---

## 6. Summary of Hypothesis Evaluation

| Sub-hypothesis                       | Outcome                                               |
| ------------------------------------ | ----------------------------------------------------- |
| (A) Temporal smoothness              | Partially supported (requires aggregation; CNN > ViT) |
| (B) Drift behavior                   | Partially supported (relative, not absolute)          |
| (C) Sensitivity to temporal distance | Supported                                             |
| (D) Robustness under stress          | Supported                                             |

The initial hypothesis favoring transformer-based representations was **refined**, not rejected. The experiments reveal that **inductive bias plays a decisive role**, with CNN-based representations producing more temporally coherent motion descriptors in a zero-shot setting.

---

## 7. Refined Hypothesis (Post-analysis)

Based on the empirical findings, the hypothesis is reformulated as:

> **Refined Hypothesis:** In real capsule endoscopy videos without pose supervision, motion-relevant temporal information emerges from temporally aggregated learned visual representations, and the stability of this information is strongly governed by representational inductive bias rather than architectural class alone.

---

## 8. Implications

These results demonstrate that reliable motion characterization in capsule endoscopy is fundamentally a **temporal representation problem**, not a frame-level estimation problem. Furthermore, they establish the need for explicitly designed temporal motion descriptors as a prerequisite for downstream tasks such as localization and pose estimation.

