"""
inference_mcmc.py  (CERT Insider Threat Edition)
=================================================
Gibbs Sampling approximate inference engine.

Supports binary and multi-state variables (e.g. RiskLevel with 3 states).

Core API
--------
  mcmc = MCMCEngine(bayes_net, factors)
  result = mcmc.sample_posterior(query_var, evidence, num_samples=10000)
  # result is a dict: {state: estimated_probability}

Experiment 3 helper
-------------------
  mcmc.convergence_study(query_var, evidence, sample_counts=[1000,5000,10000,50000])
  Returns a dict of results for each sample count.
"""

import time
import random
import numpy as np


class MCMCEngine:
    """
    Gibbs Sampling for approximate Bayesian inference.

    At each step, one non-evidence variable is sampled from its
    conditional distribution P(var | MB(var)) where MB(var) is
    the Markov blanket computed using the factor tables.
    """

    def __init__(self, bayes_net, initial_factors: list):
        self.bn       = bayes_net
        self.factors  = initial_factors

        # Pre-build: node → list of factors that involve it
        self._node_factors: dict = {n: [] for n in bayes_net.nodes}
        for f in initial_factors:
            for v in f.variables:
                self._node_factors[v].append(f)

    # ------------------------------------------------------------------
    def _eval_joint_local(self, state: dict, focus_nodes: set) -> float:
        """
        Evaluate the product of factors that involve any node in focus_nodes
        under the given state assignment.  Used to compute the conditional
        probability for Gibbs sampling.
        """
        relevant: set = set()
        for n in focus_nodes:
            for f in self._node_factors.get(n, []):
                relevant.add(id(f))   # use object id to deduplicate

        prod = 1.0
        for f in self.factors:
            if id(f) in relevant:
                key = tuple(state.get(v, 0) for v in f.variables)
                prod *= f.table.get(key, 1e-15)
        return prod

    # ------------------------------------------------------------------
    def sample_posterior(
        self,
        query_var:   str,
        evidence:    dict  = None,
        num_samples: int   = 10_000,
        burn_in:     int   = 2_000,
        seed:        int   = None,
    ) -> tuple[dict, dict]:
        """
        Estimate P(query_var | evidence) using Gibbs Sampling.

        Parameters
        ----------
        query_var   : Node to query.
        evidence    : Observed evidence {node: state_value}.
        num_samples : Number of samples to collect (after burn-in).
        burn_in     : Number of steps to discard at the start.
        seed        : RNG seed for reproducibility.

        Returns
        -------
        (posterior_dict, stats_dict)
        posterior_dict : {state_value: estimated_probability}
        """
        if evidence is None:
            evidence = {}
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        t0 = time.perf_counter()

        # Get the state space for each node
        node_states = self.bn.node_states

        # Initialise state: fix evidence, randomise everything else
        state     = {}
        free_vars = []
        for n in self.bn.nodes:
            if n in evidence:
                state[n] = evidence[n]
            else:
                states_n  = node_states.get(n, [0, 1])
                state[n]  = random.choice(states_n)
                free_vars.append(n)

        q_states = node_states.get(query_var, [0, 1])
        counts   = {s: 0 for s in q_states}
        total    = 0

        total_steps = burn_in + num_samples

        for step in range(total_steps):
            # Pick a free variable at random
            var    = random.choice(free_vars)
            var_st = node_states.get(var, [0, 1])
            mb     = {var} | self.bn.get_markov_blanket(var)

            # Compute unnormalised conditional for each state of var
            probs = []
            for s in var_st:
                state[var] = s
                probs.append(self._eval_joint_local(state, mb))

            # Normalise and sample
            total_p = sum(probs)
            if total_p > 0:
                norm_p = [p / total_p for p in probs]
            else:
                norm_p = [1.0 / len(var_st)] * len(var_st)

            chosen     = random.choices(var_st, weights=norm_p, k=1)[0]
            state[var] = chosen

            # Collect sample after burn-in
            if step >= burn_in:
                counts[state[query_var]] = counts.get(state[query_var], 0) + 1
                total += 1

        elapsed  = time.perf_counter() - t0
        n_collect = max(1, total)
        posterior = {s: counts[s] / n_collect for s in q_states}

        stats = {
            "query_var":      query_var,
            "num_samples":    num_samples,
            "burn_in":        burn_in,
            "total_steps":    total_steps,
            "runtime_sec":    elapsed,
            "samples_per_sec": n_collect / max(1e-9, elapsed),
        }

        return posterior, stats

    # ------------------------------------------------------------------
    def convergence_study(
        self,
        query_var:     str,
        evidence:      dict  = None,
        sample_counts: list  = None,
        ve_reference:  dict  = None,
        burn_in:       int   = 2_000,
        seed:          int   = 42,
    ) -> dict:
        """
        Run Gibbs sampling at multiple sample counts and compare to a VE
        reference posterior.  Used for Experiment 3.

        Parameters
        ----------
        query_var     : Node to query.
        evidence      : Observed evidence dict.
        sample_counts : List of sample counts to test (e.g. [1000,5000,10000,50000]).
        ve_reference  : Dict {state: prob} from VE (ground truth for error calc).
        burn_in       : Burn-in steps (used for every run).
        seed          : Base RNG seed.

        Returns
        -------
        Dict keyed by sample count, each containing:
          posterior, runtime_sec, l1_error (vs VE reference), samples_per_sec
        """
        if sample_counts is None:
            sample_counts = [1_000, 5_000, 10_000, 50_000]
        if evidence is None:
            evidence = {}

        results = {}
        for n in sample_counts:
            post, stats = self.sample_posterior(
                query_var,
                evidence=evidence,
                num_samples=n,
                burn_in=burn_in,
                seed=seed,
            )
            l1_error = None
            if ve_reference is not None:
                l1_error = sum(
                    abs(post.get(s, 0.0) - ve_reference.get(s, 0.0))
                    for s in ve_reference
                )
            results[n] = {
                "posterior":      post,
                "runtime_sec":    stats["runtime_sec"],
                "samples_per_sec":stats["samples_per_sec"],
                "l1_error":       l1_error,
            }
        return results

    # ------------------------------------------------------------------
    def query_state(
        self,
        query_var:   str,
        target_state,
        evidence:    dict = None,
        num_samples: int  = 10_000,
        burn_in:     int  = 2_000,
        seed:        int  = None,
    ) -> tuple[float, dict]:
        """Return P(query_var = target_state | evidence) as a scalar."""
        posterior, stats = self.sample_posterior(
            query_var, evidence=evidence,
            num_samples=num_samples, burn_in=burn_in, seed=seed,
        )
        return posterior.get(target_state, 0.0), stats
