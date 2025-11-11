Overview

This repository contains the implementation of a Feedforward Neural Network (FNN) designed for alpha signal combination and optimization in high-frequency trading environments.

The input dataset comprises multiple alpha signals derived from tick-by-tick market data, representing a wide spectrum of short-term predictive features. The objective is to efficiently combine these alphas into a single optimized signal that maximizes out-of-sample predictive performance — measured primarily through validation correlation and R² (coefficient of determination) metrics.

Unlike standard regression or neural network setups, this implementation introduces domain-specific constraints directly into the training process. These constraints guide the optimization to enforce structure and interpretability that align with the realities of high-frequency trading, such as:
	•	Sign constraints on certain weights to preserve known relationships.
	•	Feature zeroing for signals deemed uninformative or redundant based on domain knowledge.
	•	Iterative constraint satisfaction ensuring model robustness and realistic parameter dynamics.

This approach improves generalization and stability while maintaining alignment with financial domain priors, leading to higher and more consistent validation correlations compared to unconstrained models.

⸻

Key Features
	•	✅ Tick-level alpha combination using a feedforward NN.
	•	⚙️ Custom training loop with constraint-aware optimization.
	•	📊 Evaluation metrics: R², Pearson correlation, and validation scatter plots.
	•	🧠 Configurable network architecture and regularization strategies.
	•	🧩 Easily extendable framework for experimenting with additional constraints or architectures.

⸻

Use Case

This framework is particularly suited for quantitative researchers and HFT practitioners seeking to:
	•	Combine multiple short-horizon alphas into a unified predictive signal.
	•	Apply domain-informed constraints during training.
	•	Benchmark and interpret alpha combination strategies on high-frequency datasets.


Results:

  Val Loss: 60204.60156250, MSE: 60204.59920108, MAE: 169.85937865, R²: 0.0105, Corr: [np.float64(9.863369655895388), np.float64(10.511540556322839), np.float64(11.432941544194087)]
⸻
