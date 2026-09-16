"""
inference_ve.py  (CERT Insider Threat Edition)
===============================================
Variable Elimination exact inference engine.

Supports both binary nodes and multi-state nodes (e.g. RiskLevel
with states 0=Low, 1=Medium, 2=High).

Core API
--------
  ve = VariableEliminationEngine(bayes_net, factors)
  result = ve.query(query_var, evidence, strategy='min-degree')
  # result is a dict: {state: probability}

Methods
-------
  query(query_var, evidence, strategy)   → dict of posterior probabilities
  compare_strategies(query_var, evidence) → timing comparison dict
"""

import time
import copy

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


class VariableEliminationEngine:
    """
    Exact Bayesian inference via Variable Elimination.

    Supports:
    - Binary and multi-state variables.
    - Evidence restriction (Factor.condition_on_evidence).
    - Two elimination-order heuristics: min-degree, min-fill.
    - Ancestor pruning (Irrelevance Theorem).
    """

    def __init__(self, bayes_net, initial_factors: list):
        self.bn      = bayes_net
        self.factors = initial_factors

    # ------------------------------------------------------------------
    # Ancestor pruning (Irrelevance Theorem)
    # ------------------------------------------------------------------
    def _prune_ancestors(self, query_var: str, evidence: dict):
        """
        Restrict computation to factors that are ancestors of
        {query_var} ∪ evidence variables.
        Returns (pruned_factor_list, active_node_set).
        """
        targets   = set([query_var] + list(evidence.keys()))
        ancestors = self.bn.get_ancestors(targets)

        pruned = [f for f in self.factors
                  if all(v in ancestors for v in f.variables)]
        return pruned, ancestors

    # ------------------------------------------------------------------
    # Elimination order heuristics
    # ------------------------------------------------------------------
    def _get_elimination_order(
        self, vars_to_eliminate: set, active_factors: list, strategy: str
    ) -> list:
        """
        Compute an elimination ordering for hidden variables.

        strategy: 'min-degree' | 'min-fill' | 'default'
        """
        if _HAS_NX:
            # Build interaction graph from active factors
            G = nx.Graph()
            for f in active_factors:
                vlist = f.variables
                for i in range(len(vlist)):
                    for j in range(i + 1, len(vlist)):
                        G.add_edge(vlist[i], vlist[j])
        else:
            G = None   # fallback: no interaction graph

        remaining = list(vars_to_eliminate)
        order     = []

        while remaining:
            if strategy == "min-degree" and G is not None:
                best = min(remaining, key=lambda v: G.degree(v) if v in G else 0)
            elif strategy == "min-fill" and G is not None:
                def _fill_count(v):
                    if v not in G:
                        return 0
                    nbrs = list(G.neighbors(v))
                    return sum(
                        1 for i in range(len(nbrs))
                          for j in range(i + 1, len(nbrs))
                          if not G.has_edge(nbrs[i], nbrs[j])
                    )
                best = min(remaining, key=_fill_count)
            else:
                best = remaining[0]

            order.append(best)
            remaining.remove(best)

            # Update interaction graph: connect all neighbours, remove node
            if G is not None and best in G:
                nbrs = list(G.neighbors(best))
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        G.add_edge(nbrs[i], nbrs[j])
                G.remove_node(best)

        return order

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------
    def query(
        self,
        query_var:  str,
        evidence:   dict = None,
        strategy:   str  = "min-degree",
    ) -> tuple[dict, dict]:
        """
        Compute P(query_var | evidence) via Variable Elimination.

        Parameters
        ----------
        query_var : Name of the query variable.
        evidence  : Dict {var_name: observed_state_value}.
                    State values must match the representation in the CPT
                    (0/1 for binary; 0/1/2 for ternary RiskLevel).
        strategy  : Elimination order heuristic: 'min-degree' or 'min-fill'.

        Returns
        -------
        (posterior_dict, stats_dict)
        posterior_dict : {state_value: probability}  — sums to 1.0
        stats_dict     : runtime, factor sizes, induced width, etc.
        """
        if evidence is None:
            evidence = {}

        t0 = time.perf_counter()

        # Step 1: Ancestor pruning
        pruned_factors, active_nodes = self._prune_ancestors(query_var, evidence)

        # Step 2: Condition all factors on the observed evidence
        current_factors = []
        for f in pruned_factors:
            f_cond = f.condition_on_evidence(evidence)
            if len(f_cond.variables) > 0:
                current_factors.append(f_cond)

        # Identify hidden variables to eliminate
        hidden = set(active_nodes) - {query_var} - set(evidence.keys())

        # Step 3: Compute elimination order
        elim_order = self._get_elimination_order(hidden, current_factors, strategy)

        max_factor_size = 0
        induced_width   = 0

        # Step 4: Sequential elimination
        for var in elim_order:
            matching     = [f for f in current_factors if var in f.variables]
            non_matching = [f for f in current_factors if var not in f.variables]

            if not matching:
                continue

            # Multiply all matching factors into one product factor
            product = matching[0]
            for f in matching[1:]:
                product = product.multiply(f)

            # Track statistics
            max_factor_size = max(max_factor_size, len(product.table))
            induced_width   = max(induced_width, len(product.variables) - 1)

            # Marginalize (sum out) the target variable
            summed = product.marginalize(var)

            current_factors = non_matching
            if len(summed.variables) > 0:
                current_factors.append(summed)

        # Step 5: Final product over any remaining factors, then normalize
        if current_factors:
            final = current_factors[0]
            for f in current_factors[1:]:
                final = final.multiply(f)
        else:
            # Fallback: uniform over query variable states
            q_states = self.bn.node_states.get(query_var, [0, 1])
            from bayes_net import Factor
            final = Factor(
                [query_var],
                {(s,): 1.0 / len(q_states) for s in q_states},
                {query_var: q_states},
            )

        final.normalize()

        # Extract posterior distribution
        q_states = self.bn.node_states.get(query_var, [0, 1])
        posterior = {}
        for s in q_states:
            posterior[s] = final.table.get((s,), 0.0)

        elapsed = time.perf_counter() - t0

        stats = {
            "query_var":         query_var,
            "evidence_count":    len(evidence),
            "pruned_nodes":      len(self.bn.nodes) - len(active_nodes),
            "elimination_order": elim_order,
            "max_factor_size":   max_factor_size,
            "induced_width":     induced_width,
            "runtime_sec":       elapsed,
            "strategy":          strategy,
        }

        return posterior, stats

    # ------------------------------------------------------------------
    # Convenience: return just the scalar probability of one state
    # ------------------------------------------------------------------
    def query_state(
        self, query_var: str, target_state, evidence: dict = None, strategy: str = "min-degree"
    ) -> tuple[float, dict]:
        """
        Return P(query_var = target_state | evidence).

        Useful for binary nodes where you want a single scalar.
        """
        posterior, stats = self.query(query_var, evidence=evidence, strategy=strategy)
        return posterior.get(target_state, 0.0), stats

    # ------------------------------------------------------------------
    # Compare two elimination strategies
    # ------------------------------------------------------------------
    def compare_strategies(
        self, query_var: str, evidence: dict = None
    ) -> dict:
        """
        Run VE with both min-degree and min-fill and return a comparison.
        Used for Experiment 2.
        """
        results = {}
        for strat in ["min-degree", "min-fill"]:
            posterior, stats = self.query(query_var, evidence=evidence, strategy=strat)
            results[strat] = {
                "posterior":       posterior,
                "runtime_sec":     stats["runtime_sec"],
                "induced_width":   stats["induced_width"],
                "max_factor_size": stats["max_factor_size"],
            }
        return results
