"""
plot_bfgs_convergence.py
========================
Tracks and plots exactly 40 iterations of the L-BFGS-B optimization algorithm
used in CERT_35Node_BN_v3 for Maximum Likelihood parameter learning of the
Noisy-OR causal parameters.

Visualizes:
1. Loss / Objective function value (Penalized Negative Log-Likelihood) vs Iterations
2. Gradient Norm (||∇L||_2) showing convergence toward the stationary point
3. Parameter trajectory: Evolution of key causal weights & leak probabilities across 40 iterations
4. Step size / Parameter update magnitude (||θ_k - θ_{k-1}||_2)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import cert_preprocessing as P
import cert_bn_builder as B
import v3_common as C

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def run_bfgs_tracking(target_iters: int = 40):
    print("Loading CERT dataset for BFGS tracking...")
    df = C.load_base()
    R = P.FlagRules().fit(df)
    F = R.transform(df)

    learner = B.NoisyORLearner()
    E, T, cnt = learner.aggregate(F, df)
    print(f"Aggregated {len(cnt)} unique evidence-threat configurations across {len(df):,} user-days.")

    theta_history = []
    loss_history = []
    grad_norm_history = []

    # Record initial state (Iteration 0)
    th0 = learner.theta0.copy()
    f0, g0 = learner.nll_and_grad(th0, E, T, cnt)
    theta_history.append(th0.copy())
    loss_history.append(float(f0))
    grad_norm_history.append(float(np.linalg.norm(g0)))

    # scipy minimize callback passes xk (current parameter vector)
    def callback(xk):
        fk, gk = learner.nll_and_grad(xk, E, T, cnt)
        theta_history.append(xk.copy())
        loss_history.append(float(fk))
        grad_norm_history.append(float(np.linalg.norm(gk)))
        it = len(loss_history) - 1
        print(f"  BFGS Iteration {it:2d}/{target_iters} | Loss (NLL): {fk:.6f} | ||Grad||: {np.linalg.norm(gk):.6f}")

    print(f"\nStarting L-BFGS-B optimizer for {target_iters} iterations...")
    res = minimize(
        learner.nll_and_grad,
        th0,
        args=(E, T, cnt),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": target_iters, "gtol": 1e-12, "ftol": 1e-14, "disp": False}
    )

    iters_completed = len(loss_history) - 1
    print(f"L-BFGS-B reached {iters_completed} iterations. Final loss: {loss_history[-1]:.6f}\n")

    iterations = np.arange(len(loss_history))
    losses = np.array(loss_history)
    grad_norms = np.array(grad_norm_history)
    thetas = np.array(theta_history)

    # Calculate step sizes ||theta_k - theta_{k-1}||
    step_sizes = [0.0] + [float(np.linalg.norm(thetas[i] - thetas[i-1])) for i in range(1, len(thetas))]

    # Unpack parameters across iterations to observe physical probabilities
    # Let's extract:
    # 1. StagingAnomaly leak probability
    # 2. AuthAnomaly leak probability
    # 3. DataExfiltration -> USB / File Movement causal weight
    # 4. ITSabotage -> AdminTool / Port causal weight
    unpacked_params = []
    for th in thetas:
        bw, bl, tw, tl = learner.unpack(th)
        unpacked_params.append({
            "bl_Auth": bl[0],
            "bl_DataMovement": bl[2],
            "bl_Staging": bl[6],
            "tl_DataExfil": tl[0],
            "tw_Exfil": tw[0] if len(tw) > 0 else 0.5,
            "tw_Sabotage": tw[1] if len(tw) > 1 else 0.5,
        })
    p_df = pd.DataFrame(unpacked_params)

    # =========================================================================
    # 4-Panel BFGS Convergence Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"L-BFGS-B Parameter Optimization Convergence ({iters_completed} Iterations)\nNoisy-OR Causal Conditional Probability Learning on CERT DAG",
                 fontsize=14, fontweight="bold", y=0.98)

    # 1. Loss / Objective Value vs Iterations
    ax1 = axes[0, 0]
    ax1.plot(iterations, losses, "o-", color="#1f77b4", lw=2, ms=4.5, label="Penalized NLL")
    ax1.set_title("A. Objective Function Descent (Negative Log-Likelihood)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("BFGS Iteration (k)", fontsize=10)
    ax1.set_ylabel("Loss / Objective Value (f(θ))", fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Highlight initial vs final
    ax1.annotate(f"Init: {losses[0]:.4f}", xy=(0, losses[0]), xytext=(3, losses[0] + 0.04),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2), fontsize=9, fontweight="bold")
    ax1.annotate(f"Iter {iters_completed}: {losses[-1]:.4f}", xy=(iters_completed, losses[-1]),
                 xytext=(iters_completed - 12, losses[-1] + 0.08),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2), fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Gradient Norm (||∇f(θ)||_2) Log Scale
    ax2 = axes[0, 1]
    ax2.plot(iterations, grad_norms, "s-", color="#d62728", lw=2, ms=4.5, label="Gradient L2 Norm ||∇f||")
    ax2.set_yscale("log")
    ax2.set_title("B. First-Order Optimality: Gradient Norm ||∇f(θ)||_2", fontsize=11, fontweight="bold")
    ax2.set_xlabel("BFGS Iteration (k)", fontsize=10)
    ax2.set_ylabel("||∇f(θ)||_2 (Log Scale)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(grad_norms[-1], color="grey", ls=":", alpha=0.7, label=f"Final ||∇f|| = {grad_norms[-1]:.2e}")
    ax2.legend(loc="upper right", fontsize=9)

    # 3. Evolution of Causal Probabilities / Parameters
    ax3 = axes[1, 0]
    ax3.plot(iterations, p_df["bl_Auth"], label="Leak: AuthAnomaly", lw=1.8, color="#ff7f0e")
    ax3.plot(iterations, p_df["bl_DataMovement"], label="Leak: DataMovementAnomaly", lw=1.8, color="#2ca02c")
    ax3.plot(iterations, p_df["bl_Staging"], label="Leak: StagingAnomaly", lw=1.8, color="#9467bd")
    ax3.plot(iterations, p_df["tl_DataExfil"], label="Leak: DataExfiltration Threat", lw=1.8, color="#8c564b")
    ax3.set_title("C. Parameter Trajectory: Leak Probabilities Evolution", fontsize=11, fontweight="bold")
    ax3.set_xlabel("BFGS Iteration (k)", fontsize=10)
    ax3.set_ylabel("Physical Probability Value [0, 0.10]", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8.5)

    # 4. Parameter Update Step Magnitude ||θ_k - θ_{k-1}||
    ax4 = axes[1, 1]
    ax4.bar(iterations[1:], step_sizes[1:], color="#17becf", alpha=0.85, width=0.6,
            edgecolor="#0b7285", lw=0.7, label="Step Norm ||θ_k - θ_{k-1}||")
    ax4.set_title("D. Parameter Update Step Magnitude (Decay to Steady State)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("BFGS Iteration (k)", fontsize=10)
    ax4.set_ylabel("Parameter Displacement ||Δθ||_2", fontsize=10)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_plot = os.path.join(C.RESULTS, "bfgs_40_iterations_convergence.png")
    plt.savefig(out_plot, dpi=180)
    print(f"Saved convergence plot to: {out_plot}")

    # Save tracking history CSV
    hist_df = pd.DataFrame({
        "iteration": iterations,
        "nll_loss": losses,
        "grad_norm": grad_norms,
        "step_size": step_sizes,
    })
    hist_csv = os.path.join(C.RESULTS, "bfgs_convergence_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"Saved convergence tracking table to: {hist_csv}")

if __name__ == "__main__":
    run_bfgs_tracking(target_iters=40)
