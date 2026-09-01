# Preliminary Research: Multimodal Skeleton + Inertial Fusion for Medical Action Recognition

**Research Area:** Multimodal HAR / Federated Learning  
**Date:** April 2026  
**Context:** Literature survey conducted as part of Phase 4 (Multimodal Extension) of the federated skeleton-based action recognition project. This report investigates the feasibility of combining skeleton data with inertial measurement unit (IMU) data for improved medical action recognition, with particular focus on synthetic IMU generation from existing 3D skeleton data and fusion architectures compatible with our STGCN++ pipeline.

---

## 1. Motivation

Skeleton-based action recognition using graph convolutional networks (GCNs) has demonstrated strong performance for human activity recognition. However, real-world medical monitoring systems increasingly deploy **wearable inertial sensors** (accelerometers, gyroscopes) alongside or instead of vision-based skeleton estimation. The complementary nature of these modalities — skeleton captures spatial joint configuration, IMU captures local dynamics and acceleration — suggests that multimodal fusion could improve recognition accuracy, particularly for fall-related actions where acceleration signals are highly discriminative.

### Research Questions

1. **What datasets provide aligned skeleton + IMU data** for medical action recognition?
2. **Can synthetic IMU be reliably generated** from existing 3D skeleton data to bootstrap multimodal research without new data collection?
3. **Which fusion architectures** are most compatible with our STGCN++ backbone and FL pipeline?

---

## 2. Dataset Survey

We surveyed 7 datasets for skeleton + inertial data availability, focusing on medical relevance (fall detection, elderly monitoring, clinical actions).

### 2.1 Summary

| Dataset | Subjects | Actions | Skeleton | IMU | Medical Relevance | Access |
|---------|----------|---------|----------|-----|-------------------|--------|
| **UTD-MHAD** | 8 | 27 | Kinect v1 (20 joints, 3D) | 1 sensor: accel + gyro, 50Hz | Fair | Open download |
| **SmartFallMM** | 40 (24 elderly) | 14 (5 falls + 9 ADL) | Azure Kinect (3 views) | Wrist watch + hip phone, ~50Hz | **Excellent** | GitHub |
| **MMAct** | 40 | 37 | 2D keypoints (OpenPose) | Accel + gyro + orientation | Good | Request (Hitachi) |
| **C-MHAD** | 12 | 12 | Video (extractable) | Shimmer3: accel + gyro, 50Hz | **Excellent** (has falls) | Open |
| **K-Fall** | 32 | 36 (15 falls) | Video (labeling only) | 9-axis IMU, 100Hz, lower back | **Excellent** | Open |
| **Berkeley MHAD** | 12 | 11 | Optical MoCap | 6 accelerometers | Poor (exercise) | Open |
| **NTU RGB+D 60** | 40 | 60 | Kinect v2 (25 joints, 3D) | **None** | Good (medical subset) | Request |

### 2.2 Key Findings

**UTD-MHAD** is the most commonly used benchmark for skeleton+IMU research (1027 citations). It provides clean, well-synchronized Kinect skeleton and wrist/thigh-mounted inertial data. The dataset is small (861 samples) but well-studied, making it ideal for **validation** of our fusion approaches against published baselines.

**SmartFallMM** is the most medically relevant dataset identified. It uniquely includes **elderly subjects aged 60–93** performing falls and activities of daily living, with both Azure Kinect skeleton and multiple wearable sensors. A 2025 publication by the dataset authors (Debnath et al.) demonstrated that synthetic IMU generated from skeleton video is viable for fall detection, directly validating our planned approach.

**NTU RGB+D 60** (our current dataset) contains rich 3D skeleton data from Kinect v2 (25 joints) but **no companion IMU data**. However, the availability of 3D joint trajectories makes it the strongest candidate for **synthetic IMU generation** — the approach we adopt as our primary strategy.

---

## 3. Synthetic IMU Generation from 3D Skeleton

### 3.1 Literature Review

We identified 6 methods for generating virtual IMU signals from motion data:

| Method | Source | Approach | Availability |
|--------|--------|----------|-------------|
| **Physics-based (direct)** | SmartFallMM/Debnath 2025 | Discrete 2nd derivative of joint position | Trivial to implement |
| IMUTube | IMWUT 2020 (216 citations) | Video → 2D pose → 3D lift → virtual accel | Licensed (Georgia Tech) |
| IMUGPT 2.0 | UbiComp 2024 (60 citations) | Text → motion synthesis → virtual IMU | Open |
| SynHAR | IEEE Access 2024 | MoCap → human surface model → IMU | Open |
| Virtual IMU Platform | IEEE Trans. 2025 | Open-source MoCap → customizable virtual IMU | Open source |
| Wonderwall | IMWUT 2026 | Foundation model: MoCap → virtual accel | Likely available |

### 3.2 Selected Approach: Physics-Based Simulation

The most practical and well-validated approach for our setup is **direct physics-based conversion** from 3D joint trajectories. Given a joint position $p(t) = [x(t), y(t), z(t)]$ at frame $t$ in the NTU 3D skeleton:

$$a(t) = \frac{p(t+1) - 2p(t) + p(t-1)}{\Delta t^2}$$

where $\Delta t = 1/30$ seconds (Kinect v2 frame rate).

This discrete second derivative yields acceleration in m/s², directly analogous to accelerometer readings. Post-processing includes:

1. **Low-pass filtering** (2nd-order Butterworth, ~12 Hz cutoff) to smooth quantization artifacts from the discrete derivative
2. **Gaussian noise injection** ($\sigma \approx 0.05$ m/s²) to simulate real sensor noise characteristics
3. **Temporal resampling** to match the target clip length (100 frames, matching our skeleton input)
4. **Normalization** to [-1, 1] range for neural network input

### 3.3 Virtual Sensor Placement

We map virtual IMU sensors to specific NTU-25 skeleton joints:

| Virtual Sensor | NTU Joint Index | Joint Name | Clinical Rationale |
|---------------|----------------|------------|-------------------|
| **Right wrist** | 11 | Right hand | Captures hand gestures (touch head/chest/neck actions) |
| **Waist/hip** | 0 | Base of spine | Captures gait and fall dynamics |

This yields a **6-channel** synthetic IMU signal (2 sensors × 3 acceleration axes) per sample, providing complementary motion information to the spatial skeleton representation.

### 3.4 Validation

The physics-based approach is validated by multiple recent publications:
- **SmartFallMM** (Debnath et al., 2025) demonstrated skeleton-to-IMU conversion for fall detection with elderly subjects
- **IMUTube** (Kwon et al., IMWUT 2020, 216 citations) established that virtual IMU from video-derived skeletons produces competitive HAR performance
- **SynHAR** (Uhlenberg et al., IEEE Access 2024) showed synthetic IMU achieves within 5% of real IMU accuracy for activity recognition

---

## 4. Fusion Architecture Survey

We evaluated 5 multimodal fusion approaches for combining skeleton and IMU features:

### 4.1 Summary

| Approach | Key Idea | Reported Gain | Complexity | Compatibility |
|----------|----------|---------------|------------|---------------|
| **Fusion-GCN** | IMU as extra nodes/channels in GCN graph | +12.4% F1 | Low | **Excellent** |
| **Late Fusion** | Dual-stream: STGCN++ + 1D-CNN → concat | +5–15% F1 | Low | **Excellent** |
| **Cross-Attention** | Each modality attends to the other | +5–10% | Medium | Good |
| Knowledge Distillation | Train with both, distill to single modality | Compact model | Medium | Good |
| Adversarial Cross-Modal | IMU↔Skeleton translation network | Competitive | High | Poor (withdrawn paper) |

### 4.2 Selected Approaches

#### Approach 1: Late Fusion (Baseline)

The simplest multimodal architecture, serving as our primary baseline:

- **Skeleton stream:** Existing STGCN++ backbone → 256-dimensional feature vector
- **IMU stream:** Lightweight 1D-CNN (Conv1d → BN → ReLU → Conv1d → BN → ReLU → Global Average Pool) → 128-dimensional feature vector
- **Fusion:** Concatenate [256 + 128 = 384] → Linear(384, 10) → classification

This requires **no modifications** to the existing STGCN++ model — the IMU branch is a simple add-on. The 1D-CNN encoder adds ~50K parameters to the ~443K skeleton model.

**References:** C-MHAD (Wei et al., Sensors 2020); Late fusion baselines in UTD-MHAD literature.

#### Approach 2: Fusion-GCN (Primary Method)

Integrates IMU data directly into the graph convolutional network by treating IMU sensors as additional nodes or channels in the skeleton graph:

- **Channel fusion:** Expand input from $(N, 2, 100, 17, 3)$ to $(N, 2, 100, 17, 9)$ — append 6 IMU channels (mapped to corresponding joints, zero-pad others)
- **Node fusion:** Add 2 new virtual nodes to the 17-joint COCO graph (19 total), update the adjacency matrix, populate new nodes with IMU features

This approach leverages the existing GCN architecture to learn joint spatial-temporal-inertial features in a unified framework. Duhme et al. (GCPR 2021) reported **+12.4% F1 improvement** on MMAct with this strategy.

**References:** Fusion-GCN (Duhme et al., arXiv:2109.12946, GCPR 2021).

#### Approach 3: Cross-Attention Fusion (Future Work)

More sophisticated approach where each modality attends to the other via cross-attention layers before final classification. Expected to capture complementary information more effectively than simple concatenation, but requires more compute and is harder to tune. Planned for investigation after the baseline approaches are validated.

**References:** Han et al., IEEE Sensors Journal 2026.

---

## 5. Proof-of-Concept Experiment Design

### 5.1 Implemented (Quick Scan)

We implemented the Late Fusion approach in our experiment notebook (`train_colab_methods.ipynb`) alongside the FL methods comparison:

1. **Synthetic IMU generation** from NTU RGB+D 60 3D skeleton data (25 joints) for all 10 medical classes
2. **Late Fusion model**: STGCN++(skeleton, 256-d) + 1D-CNN(IMU, 128-d) → FC(384, 10)
3. **Comparison**: Centralized skeleton-only vs. centralized late fusion, both trained for 10 epochs on T4
4. **Dataset**: Same medical subset, samples with both 2D HRNet and 3D Kinect skeleton data

### 5.2 Planned Extensions

| Experiment | Description | Estimated Effort |
|------------|-------------|-----------------|
| Fusion-GCN (channel) | IMU channels added to STGCN++ input at corresponding joints | Modify input dims + adjacency |
| Fusion-GCN (node) | Virtual IMU nodes in skeleton graph | New adjacency matrix, 19-joint graph |
| FL + Multimodal | Run FedAvg/FedProx on late fusion model | Extend FL loop to multi-input model |
| Real IMU validation | Train on UTD-MHAD with real skeleton + IMU | New data loader, 20-joint graph |

---

## 6. Feasibility Assessment

### Can synthetic IMU from NTU skeleton bootstrap useful multimodal research?

**Yes.** The approach is validated by peer-reviewed publications (IMUTube, SmartFallMM, SynHAR) and requires no additional data collection. The implementation is straightforward — a single function converts 3D joint trajectories to acceleration via discrete differentiation.

**Potential limitations:**
- Synthetic IMU lacks sensor-specific noise characteristics (bias drift, vibration artifacts)
- No gyroscope signal (only accelerometer) from position-based derivatives
- Acceleration derived from 30 fps skeleton may miss high-frequency dynamics captured by 50-100 Hz real IMU

**Mitigation:** Results on synthetic IMU establish a lower bound. If fusion improves accuracy with synthetic data, real IMU data (from UTD-MHAD or SmartFallMM) will likely produce larger gains. The synthetic approach serves as a **proof of concept** to justify pursuing real multimodal datasets.

### Recommended Next Steps

1. **Run the quick-scan experiment** to determine if synthetic IMU provides measurable accuracy improvement over skeleton-only
2. If positive: implement **Fusion-GCN** for tighter integration, then combine with FL methods
3. If marginal: download **UTD-MHAD** real data to determine if real IMU performs better than synthetic
4. Long-term: request access to **SmartFallMM** (most medically relevant, elderly subjects, fall-specific)

---

## 7. Key References

| Paper | Venue | Year | Relevance |
|-------|-------|------|-----------|
| UTD-MHAD (Chen et al.) | ICIP | 2015 | Gold standard skeleton+IMU benchmark (1027 cites) |
| SmartFallMM (Debnath et al.) | Sensors | 2025 | Elderly fall detection, skeleton→IMU validation |
| MMAct (Kong et al.) | ICCV | 2019 | Large-scale multimodal HAR (7 modalities) |
| C-MHAD (Wei et al.) | Sensors | 2020 | Falls + late fusion baseline |
| K-Fall (Yu et al.) | — | — | 32-subject fall dataset, IMU-focused |
| Fusion-GCN (Duhme et al.) | GCPR | 2021 | GCN-based skeleton+IMU fusion (+12.4% F1) |
| IMUTube (Kwon et al.) | IMWUT | 2020 | Video→virtual IMU pipeline (216 cites) |
| IMUGPT 2.0 (Leng et al.) | UbiComp | 2024 | Text→motion→virtual IMU |
| SynHAR (Uhlenberg et al.) | IEEE Access | 2024 | MoCap→synthetic IMU validation |
| Wonderwall (2026) | IMWUT | 2026 | Foundation model for virtual accelerometer |
| Multimodal Wearable HAR Survey (Ni et al.) | arXiv:2404.15349 | 2024 | Comprehensive survey (43 cites) |
| Medical Skeleton+IMU Fusion (Han et al.) | IEEE Sensors J. | 2026 | Skeleton global + IMU local limb features |
