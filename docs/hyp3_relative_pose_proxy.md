# Hypothesis 3: Regime-Aware Relative Pose Proxy for Capsule Endoscopy

## Hypothesis Statement

**A regime-aware relative pose proxy**, constructed by selectively accumulating visual motion estimates during stable temporal regimes, produces a more stable, interpretable, and drift-controlled approximation of relative capsule progression than naïve continuous accumulation.

This hypothesis evaluates whether **temporal regime abstraction** can be used to actively condition motion integration in a manner that satisfies core properties required for pose estimation in real capsule endoscopy (WCE) videos.

---

## Motivation

Absolute localization in capsule endoscopy requires reliable pose estimation over long temporal horizons. However, real-world WCE videos exhibit significant noise:

* **Intermittent and non-uniform motion.**
* **Non-rigid tissue deformation.**
* **Visual artifacts:** Bubbles, fluids, and specular highlights.
* **Prolonged periods** of unreliable or ambiguous visual motion.

**The Problem:** Naïvely accumulating frame-to-frame motion estimates under these conditions leads to rapid drift, unstable pose trajectories, and spurious oscillations.

---

## Definition: Relative Pose Proxy

A **relative pose proxy** is a temporally accumulated signal that approximates progression through the environment. It assumes that meaningful motion occurs only during stable motion regimes.

Rather than estimating physical pose or metric displacement, the proxy exhibits behavioral properties consistent with pose evolution while remaining agnostic to absolute scale and orientation.

> **Formal Logic:** Motion accumulation is gated by a learned temporal regime indicator. Updates occur only when the system is in a regime deemed **stable**.

### Desired Properties of a Pose-Like Signal

To meaningfully resemble relative pose progression, the signal should satisfy:

1. **Directional Consistency:** The signal should not exhibit frequent spurious reversals.
2. **Piecewise Smoothness:** Pose should evolve smoothly during reliable motion and plateau during unreliable periods.
3. **State-Dependent Updates:** Updates must be conditioned on signal reliability rather than being continuous.
4. **Drift Control:** Suppression of spurious motion contributions to prevent long-term error accumulation.

---

## Experimental Setup

* **Visual Representation:** CNN embedding distance (ResNet-18, ImageNet pretrained).
* **Motion Signal:** Frame-to-frame embedding change.
* **Temporal Regimes:** Learned via unsupervised clustering ().
* **Stable Regime:** The dominant regime identified during abstraction.

### Accumulation Variants

| Variant | Description |
| --- | --- |
| **Naïve** | Continuous accumulation across all frames. |
| **Regime-aware** | Accumulation gated by the identified stable regime. |

---

## Results & Analysis

### Experiment 1: Velocity Stability

**Question:** Does the proxy produce stable velocity estimates during motion?
**Method:** Velocity is defined as the first temporal difference of the accumulated signal. Conditional velocity variance is computed only over active update steps.

* **Conditional velocity variance (Naïve):** 
* **Conditional velocity variance (Regime-aware):** 
* **Stability Improvement Factor:** ****

**Interpretation:** The regime-aware proxy produces substantially more stable updates. This aligns with the requirements of downstream state estimators (e.g., Kalman filters) which require predictable motion increments.

---

### Experiment 2: Update Selectivity (Drift Suppression)

**Question:** Does the proxy suppress spurious updates that induce oscillations?
**Method:** Evaluation of **update sparsity** (the fraction of time steps where the proxy updates).

* **Naïve Update Rate:** 
* **Regime-aware Update Rate:** 

**Interpretation:** By intentionally suppressing updates during unstable regimes, the system prevents the accumulation of unreliable motion, directly reducing drift.

---

### Experiment 3: Regime Alignment

**Question:** Does pose progression align with stable motion regimes?
**Method:** Measuring the fraction of proxy updates occurring within the dominant stable regime.

* **Naïve Alignment:** 
* **Regime-aware Alignment:** 

**Interpretation:** This confirms that the accumulation is genuinely state-conditioned. Updates occur almost exclusively during periods of reliable motion.

---

## Comparative Evaluation: Optical Flow vs. CNN Embeddings

To assess if regime-awareness alone is sufficient, we applied the same logic to **Optical Flow** magnitude signals.

**Key Finding:**
While regime-aware gating enforces selective integration for both, **CNN embedding-based proxies** exhibit significantly higher conditional stability. Optical flow proxies remained unstable even within "stable" regimes.

> **Conclusion:** Regime-awareness must be paired with a **semantically meaningful** motion representation to yield reliable pose-like behavior.

---

## Summary of Findings

Across all experiments, the regime-aware relative pose proxy demonstrates:

* **Selective motion integration** based on temporal stability.
* **Substantially improved stability** of pose increments.
* **Strong alignment** between pose updates and stable motion regimes.
* **Effective suppression** of spurious drift accumulation.

### Final Conclusion

**Hypothesis 3 is supported.** The regime-aware relative pose proxy provides a principled intermediate representation between raw visual signals and full pose estimation. This establishes regime-aware motion gating as a necessary foundational component for future sensor-fusion or SLAM-based localization pipelines in capsule endoscopy.

