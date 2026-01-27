Phase 1 Summary: Motion Cues in Real Capsule Endoscopy Video

**Video ID:** `2f513ad4ee5e4630`
**Objective:** Evaluate whether classical and learned frame-to-frame signals can reliably reflect capsule motion in real gastrointestinal environments.


## 1. Qualitative Ground Truth (Human Observation)

Manual inspection of the capsule endoscopy video reveals that capsule motion is **rarely clean or visually explicit**. Long intervals exhibit slight forward drift with minimal appearance change, while other segments contain strong visual changes caused by **fluid motion, bubbles, folds, debris, and sensor artifacts**. Motion is often inferred only by **temporal context** (e.g., gradual parallax, fold evolution), not by instantaneous frame differences .

This establishes an important baseline: **visual change and physical motion are frequently decoupled** in real capsule footage.


## 2. Pixel Difference: What Failed

Mean absolute pixel difference was evaluated as a naive motion proxy. The signal was found to be:

* Highly sensitive to **bubbles, fluid, lighting changes, and color shifts**
* Prone to **large spikes during non-motion events**
* Prone to **collapse during smooth forward motion**
* Temporally inconsistent, often remaining elevated after motion regimes ended

For example, during smooth forward traversal through homogeneous regions, pixel difference frequently dropped to low values, while sudden appearance of folds or fluid caused extreme spikes unrelated to translation. Overall, pixel difference behaves as an **appearance-change detector**, not a motion signal, and fails to encode motion stability, direction, or continuity .


## 3. Optical Flow: Partial Improvement, Persistent Failure

Dense optical flow (Farneback) improved sensitivity to **structural transitions**, such as entering folded regions or new anatomical contexts. However, it still exhibited fundamental limitations:

* Overreacted to **fluid dynamics and independently moving particles**
* Responded strongly to **sensor artifacts (e.g., red static)**
* Failed to reliably represent **smooth forward translation**
* Did not encode **motion direction or stability**

While optical flow captured local gradients more effectively than pixel difference, it remained dominated by **appearance deformation rather than camera motion**, especially in fluid-dominated scenes. Thus, optical flow magnitude is also not a reliable standalone proxy for capsule motion .



## 4. CNN Embeddings: What Improved

A frozen ImageNet-pretrained ResNet-18 was used to extract per-frame embeddings, with motion approximated as the L2 distance between consecutive embeddings. Compared to classical baselines, CNN embeddings showed clear improvements:

* **No collapse to zero** during smooth motion
* **Bounded, stable signal range**, avoiding extreme spikes
* Better alignment with **semantic scene transitions** (e.g., entering new regions)
* Reduced sensitivity to low-level noise compared to pixel difference and optical flow

Importantly, embedding distance reflected **scene representation change**, not raw appearance change. Oscillatory patterns corresponded well to subtle drift, rotations, and peristaltic motion observed qualitatively. However, embeddings still lacked directionality and could not distinguish translation from rotation or deformation on their own .



## 5. What Is Still Missing

Across all methods, a consistent limitation emerged:

**Frame-to-frame signals, even learned ones, cannot reliably encode motion direction, displacement, or stability in isolation.**

CNN embeddings improve robustness but remain **memoryless**. They capture *what* changes, not *how* it evolves over time. Motion in capsule endoscopy is fundamentally a **temporal inference problem**, requiring aggregation and consistency over multiple frames.


## 6. Phase-1 Conclusion

This phase demonstrates that:

* Classical motion cues (pixel difference, optical flow) fail systematically on real capsule data
* Learned visual representations provide a **more stable and semantically meaningful signal**
* However, **temporal modeling is essential** to convert scene change into motion understanding

This motivates Phase-2 work focusing on **temporal representations**, including Transformer-based embeddings and temporal aggregation.

