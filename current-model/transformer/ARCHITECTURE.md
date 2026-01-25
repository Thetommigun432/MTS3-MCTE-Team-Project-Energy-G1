# NILMFormer Architecture (Detailed)

## 1. High-Level Concept
**NILMFormer** is a hybrid "Seq2Point" model designed for Single-Appliance Load Monitoring. It combines the local feature extraction of **Dilated Convolutions** with the long-range dependency modeling of **Transformers**, augmented by a specialized **Stationarization** mechanism to handle the non-stationary nature of power grid data.

### Key Innovation: Stationarization
The core problem in NILM is that power signals vary wildly between households (different voltage baselines, different background noise). NILMFormer solves this by:
1.  **Normalizing** the input window (Mean=0, Std=1).
2.  **Encoding** the original Mean/Std as a special "Stats Token".
3.  **Processing** the normalized data through the Transformer.
4.  **De-normalizing** the output using the *original* Mean/Std to recover valid Watt values.

---

## 2. Data Flow Pipeline

The model processes a sequence of electrical power readings (e.g., 512 samples at 1Hz) to predict the power consumption of a target appliance at the *midpoint* of that window.

### Step 1: Input Preprocessing
*   **Input**: Tensor of shape `(Batch, Sequence_Length, Features)`
    *   `Features[0]`: Aggregate Active Power (Main Meter).
    *   `Features[1:]`: Temporal Embeddings (Hour Sin/Cos, Day Sin/Cos, etc.).
*   **Separation**:
    *   `Load`: The aggregate power curve.
    *   `Exogenous`: Time/Day features.

### Step 2: Global Stationarization (Instance Norm)
Before any learning, the `Load` curve is normalized per-instance to remove shift and scaling issues.
```python
# Formulas
μ = mean(Load)
σ = std(Load)
Load_Norm = (Load - μ) / σ
```
*Crucially, μ and σ are saved for later.*

### Step 3: Feature Embedding (CNN)
*   **Dilated Convolutional Block**: The normalized load passes through a stack of ResNet-like 1D Convolutions with increasing dilation factors `[1, 2, 4, 8]`.
    *   *Purpose*: Captures local patterns (edges, transient states) and expands the receptive field efficiently.
*   **Exogenous Projection**: Time features are projected linearly and concatenated.
*   **TokenStats**: The original `μ` and `σ` are concatenated and projected into a vector vector `Stats_Token`. This is appended to the sequence so the Transformer "knows" the original scale.

### Step 4: Transformer Encoder (The Brain)
*   **Structure**: Standard Transformer Encoder layers.
*   **Mechanism**: Self-Attention allows every point in the sequence to attend to every other point.
*   **Context**: For a 512-window, the model can relate an event at t=0 to an event at t=511, understanding complex cycles.

### Step 5: Seq2Point Extraction
Instead of predicting the whole sequence, we focus on the **midpoint** (`t=256`). Determining the appliance state at the exact center uses the full past and future context visible in the window.
```python
Midpoint_Feature = Transformer_Output[:, 256, :]
```

### Step 6: Dual-Head Decoding (Classification + Regression)
The midpoint feature splits into two parallel tasks:

#### A. Classification Head (State)
*   **Role**: Decides "Is the appliance ON or OFF?".
*   **Output**: `State_Logit` (passed through Sigmoid → Probability).
*   **Loss**: Binary Cross Entropy (BCE).

#### B. Regression Head (Power)
*   **Role**: Estimates "How many Watts?".
*   **Output**: `Power_Raw`. *Note: This value is in the Normalized space!*
*   **Loss**: MSE (Mean Squared Error).

### Step 7: De-Stationarization (The Fix)
We must convert `Power_Raw` back to Watts.
**Critical Logic**: We assume the appliance power scales similarly to the aggregate power.
```python
Power_Denorm = (Power_Raw * σ) + μ
Power_Denorm = ReLU(Power_Denorm) # Enforce non-negative physics
```

### Step 8: ON/OFF Gating (Conditioning)
Final output logic combines the two heads to eliminate "Ghost Load" (predicting small power when device is OFF).
```python
Final_Power = Power_Denorm * Sigmoid(State_Logit)
```
If the Classification Head says "OFF" (prob ≈ 0), the power is forced to 0.

---

## 3. Configuration Summary (Current)

| Component | Setting | Reason |
| :--- | :--- | :--- |
| **Window Size** | `512` | Optimized for Colab memory; covers ~8.5 mins at 1Hz. |
| **d_model** | `96` or `192` | Reduced parameter count (Tiny Transformer) to prevent overfitting on small data. |
| **n_layers** | `3` | Sufficient depth for feature extraction without bloat. |
| **Heads** | `4` | Balanced attention mechanism. |
| **Loss Weights** | `BCE=0.1, MSE=2.0` | Prioritizes accurate power estimation over noisy ON/OFF labeling. |

## 4. Why this beats standard CNNs (Theory)
While a CNN looks at fixed local patterns, the **NILMFormer**:
1.  **Adaptive Context**: Can ignore irrelevant noise sections via Attention weights.
2.  **Scale Invariance**: Thanks to Stationarization, it works equally well on a home with 3kW baseline and one with 10kW baseline.
3.  **State Awareness**: The explicit Classification Head acts as a gatekeeper, reducing False Positives (a common CNN failure mode).
