# Experimental Protocol for Regime-Aware Motion Integration

## 1. Problem Setting
Capsule endoscopy videos exhibit intermittent and unreliable motion, violating the assumptions of continuous motion used in classical localization pipelines. 

In this work, we evaluate **how different motion integration strategies behave under unreliable temporal dynamics.** Instead of relying on ground-truth pose (which is unavailable), we design a self-supervised evaluation protocol based on temporal consistency.

---

## 2. Dataset
We use unlabelled capsule endoscopy videos from the Kvasir-Capsule Dataset.

**Data Characteristics:**
* Long monocular video sequences
* Non-rigid environment
* Frequent motion interruptions (pauses, jitter, fluid occlusion)

**Preprocessing:**
* **Frame sampling:** 2 FPS
* **Resolution:** 224 × 224
* All sequences processed uniformly

**Sequence Selection:**
* Multiple videos from: `Datasets/kvasir-capsule/videos-raw/unlabelled_videos/`
* Evaluation is performed across multiple sequences to ensure robustness.

---

## 3. Motion Representation
We evaluate two independent motion sources. **Note:** Motion representations are not mixed; each is evaluated independently.

### 3.1 CNN Embedding Motion
* **Source:** Extract embeddings using ResNet-18 (`experiments/cnn_embedding/cnn_embedding.py`)
* **Computation:** Compute temporal differences:
$$\Delta e_t = e_t - e_{t-1}$$
* **Projection:** Project using PCA:
$$\Delta z_t \in \mathbb{R}^2$$

### 3.2 Optical Flow Motion
* **Source:** Dense optical flow using Farneback Optical Flow (`experiments/optical_flow/farneback_flow.py`)
* **Extraction:** Extract motion vector per frame (Mean or median flow: dx, dy)

---

## 4. Regime Modeling
We model motion reliability using a Hidden Markov Model (`experiments/HMM/`).

**Inputs:**
Temporal Motion Descriptor (TMD) from `experiments/tmd/compute_tmd.py`. Features include:
* Mean motion
* Variance
* Temporal consistency

**Outputs:**
Regime posterior (`results/regime_modeling/cnn_hmm/posterior.npy`), interpreted as:
$$\alpha_t = P(\text{stable regime at time } t)$$

---

## 5. Motion Integration Strategies
All methods operate on the same input motion signals.

**5.1 Raw Integration (Baseline)**
$$z_t = z_{t-1} + \Delta z_t$$

**5.2 Moving Average Smoothing**
$$\Delta z_t^{smooth} = \frac{1}{k} \sum_{i=0}^{k-1} \Delta z_{t-i}$$

**5.3 Hard Gating**
$$z_t = z_{t-1} + r_t \cdot \Delta z_t$$
> Where $r_t \in \{0,1\}$ based on a predefined threshold.

**5.4 Random Gating (Control Baseline)**
$$z_t = z_{t-1} + \tilde{r}_t \cdot \Delta z_t$$
> *Purpose: This ensures improvements are not due to arbitrary masking.*

**5.5 Regime-Aware Soft Gating (Proposed)**
$$z_t = z_{t-1} + \alpha_t \cdot \Delta z_t$$
> Where $\alpha_t$ is derived from the HMM posterior.

---

## 6. Temporal Evaluation Windows
We evaluate consistency across multiple time scales, defined by $k$:
$$k \in \{5, 10, 20, 40\}$$
* **Small $k$:** Local consistency
* **Large $k$:** Long-term drift

---

## 7. Evaluation Protocol
For each sequence, motion source, integration method, and window size ($k$), we compute metrics over sliding windows.

---

## 8. Evaluation Metrics

### 8.1 Cycle Consistency Error (Primary Metric)
We define temporal consistency via forward-backward reconstruction.

* **Forward integration:**
$$z_{t+k} = z_t + \sum_{i=1}^{k} \Delta z_{t+i}$$
* **Backward reconstruction:**
$$\hat{z}_t = z_{t+k} + \sum_{i=1}^{k} \Delta z_{t+k-i}^{back}$$
* **Metric Formulation:**
$$CCE_t^{(k)} = | z_t - \hat{z}_t |$$

**Aggregation:** Mean CCE, Median CCE, and Standard Deviation.

### 8.2 Drift Magnitude
$$D_t^{(k)} = | z_{t+k} - z_t |$$

### 8.3 Motion Smoothness
$$S_t = | \Delta z_t - \Delta z_{t-1} |$$

### 8.4 Trajectory Spread
* Covariance of $z_t$
* Eigenvalue ratio
* *Purpose: Measures structural coherence vs. chaotic drift.*

---

## 9. Baseline Comparison Strategy

| Method | Description |
| :--- | :--- |
| **Raw** | No gating |
| **Smooth** | Moving average |
| **Hard** | Threshold gating |
| **Random** | Random masking |
| **Proposed** | HMM soft gating |

**Cross-Modal Validation:** The exact same methods are applied independently to both CNN motion and Optical Flow motion.

---

## 10. Visualization
We include the following visual analyses:
1.  **Trajectory Plots:** Compare all methods and overlay trajectories.
2.  **Regime Visualization:** Plot posterior probabilities over time.
3.  **Failure Cases:** Highlight drift in raw integration vs. stability in regime-aware integration.

---

## 11. Experimental Controls
To ensure absolute fairness across all tests:
* Same motion input across all methods.
* Same initialization:
$$z_0 = 0$$
* Same hyperparameters across sequences.
* **Critical Control:** Inclusion of the random gating baseline.

---

## 12. Expected Outcome
If the hypothesis holds, we expect the following observations:
* **Raw integration:** High drift
* **Hard gating:** Discontinuities
* **Random gating:** Unstable
* **Proposed method:** Lowest inconsistency

**Key Claim:** Regime-aware motion integration reduces temporal inconsistency under intermittent dynamics.