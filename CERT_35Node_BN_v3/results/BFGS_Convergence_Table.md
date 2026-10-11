# L-BFGS-B Parameter Optimization Convergence Table

Optimization of Noisy-OR causal parameters and leak rates over the 35-Node CERT Bayesian Network structure.
Loss function corresponds to penalized negative log-likelihood with regularized prior centering.

| Iteration ($k$) | Objective Loss ($\text{NLL}$) | Gradient Norm ($\|\nabla \mathcal{L}\|_2$) | Step Size ($\|\Delta \theta\|_2$) | Optimization State |
| :---: | :---: | :---: | :---: | :--- |
| **0 (Init)** | $1.143696$ | $0.123845$ | $0.0000$ | Prior / Expert Initialization |
| **1** | $0.615548$ | $0.086229$ | $0.7812$ | Initial Steepest Descent Phase |
| **5** | $0.019889$ | $0.001549$ | $0.2140$ | Curvature Hessian Estimation |
| **10** | $0.015072$ | $0.000122$ | $0.0825$ | Fine Causal Weight Tuning |
| **20** | $0.014133$ | $0.000056$ | $0.0112$ | Asymptotic Convergence |
| **30** | $0.014113$ | $0.000001$ | $0.0018$ | Numerical Zero Gradient |
| **40** | **$0.014113$** | **$< 10^{-6}$** | **$0.0001$** | **Full Convergence Reached** |

### Mathematical Definitions
- **Objective Function:** $\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log P(T_i \mid E_i; \theta) + \lambda_w \|\theta_w - \theta_{expert}\|^2 + \lambda_{leak} \|\theta_{leak} - \text{logit}(q_0)\|^2$
- **Analytic Gradient:** Computed via Fisher identity and exact marginalization over latent behavioral indicators ($2^9 = 512$ configurations per unique evidence pattern).
- **Termination Criterion:** $\|\nabla \mathcal{L}\|_2 \le 10^{-6}$ (satisfied at iteration 35–40).
