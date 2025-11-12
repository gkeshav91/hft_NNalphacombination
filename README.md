# 🧠 Alpha Signal Combination using Feedforward Neural Network  

**Domain:** High-Frequency Trading (HFT)  
**Exchange:** NSE (Nifty Options)  
**Objective:** Alpha Optimization under Domain Constraints  

---

## 📘 Overview  

This repository implements a **Feedforward Neural Network (FNN)** designed to **combine multiple short-horizon alpha signals** into a **single optimized predictive alpha** for high-frequency trading applications.  

The dataset comprises **hundreds of alpha signals**, each derived from **tick-by-tick market data of NIFTY Options** on the **NSE**. At each timestamp, the dataset contains a high-dimensional alpha feature vector, with total samples exceeding **10 million rows**.  

The goal is to learn an optimal linear–nonlinear combination of these alphas that maximizes **out-of-sample predictive performance**, measured using **validation correlation** and **R² (coefficient of determination)**.

---

## ⚙️ Methodology  

Unlike conventional regression or neural network training setups, this implementation incorporates **domain-specific constraints** directly into the training loop to reflect the **realistic behavior of high-frequency alphas**.  

### 🔩 Constraint-Aware Optimization  

During training, the model enforces financial priors by adding **weight-based penalties** on the **first-layer parameters** of the neural network. These domain-informed constraints include:

- **Sign constraints:** Certain features are known (from domain knowledge) to have strictly positive or negative effects on the target.  
- **Pair constraints:** Pairs of features are expected to offset each other (i.e., sum to zero) due to structural symmetry in the market microstructure.  
- **Zeroing constraints:** Features identified as redundant or uninformative are penalized to drive their weights toward zero.  

These constraints are **added as auxiliary loss components**, alongside the standard prediction loss. The resulting total loss dynamically balances predictive accuracy and constraint satisfaction, leading to more **stable and interpretable alpha combinations**.

---

## 🧮 Architecture  

- **Model:** Shallow Feedforward Neural Network (customizable hidden layers and activation functions)  
- **Input Dimension:** 100s of alphas per tick (10M+ samples)  
- **Training Objective:** Weighted sum of prediction loss + constraint loss  
- **Evaluation Metrics:**  
  - Validation R²  
  - Pearson correlation between predicted and target alphas  
  - MSE and MAE  

---

## 📊 Results (Sample Run)  

| Metric | Value |
|:--|--:|
| Validation Loss | 60204.60 |
| MSE | 60204.59 |
| MAE | 169.86 |
| R² | 0.0105 |
| Validation Corr (3 splits) | 9.86, 10.51, 11.43 |

> The introduction of domain constraints led to improved validation correlation and generalization stability compared to unconstrained baselines.

---

## 🧩 Key Features  

- ✅ **Tick-level alpha combination** using a custom FNN  
- ⚙️ **Constraint-aware optimization** for realistic alpha behavior  
- 📈 **Validation metrics and scatter plot visualization** for interpretability  
- 🧠 **Easily extendable** to include new types of constraints or architectures  
- 🔍 **Configurable training setup** (batch size, regularization, hidden units)  

---

## 💼 Use Case  

This framework is designed for **quantitative researchers and high-frequency trading practitioners** aiming to:  

- Combine large sets of alpha signals into a single composite alpha  
- Enforce domain-informed structure during model training  
- Benchmark, visualize, and interpret alpha combination strategies on large-scale market datasets  

---

## 🧰 Future Extensions  

- Add support for **nonlinear feature interaction constraints**  
- Integrate **temporal models (e.g., LSTM, Transformer)** for sequential dependencies  
- Automated constraint discovery via **feature importance clustering**  

---

