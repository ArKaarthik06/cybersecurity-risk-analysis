"""
bayes_net.py  (CERT Insider Threat Edition)
============================================
Core Bayesian Network data structures and exact/approximate inference utilities.

Contains:
  - Factor        : CPT factor representation, multiplication, marginalization
  - BayesianNetwork : DAG representation, multi-state support, DAG validation

These classes are dataset-agnostic; the CERT-specific 18-node topology
is assembled in cert_bn_builder.py.

Architecture supported (18 nodes, CERT R4.2):
  Layer 1 — Observable Evidence     (10 binary nodes)
  Layer 2 — Behavioral Indicators   (5  binary nodes)
  Layer 3 — Threat Hypotheses       (2  binary nodes)
  Layer 4 — Risk Level              (1  ternary node: Low/Medium/High)
"""

import itertools
import numpy as np


# ===========================================================================
# Factor
# ===========================================================================

class Factor:
    """
    A conditional probability factor over a set of discrete variables.

    Variables can be binary (states = [0, 1]) or multi-state.

    Parameters
    ----------
    variables : list[str]
        Ordered list of variable names covered by this factor.
    table : dict
        Maps assignment tuples to float probabilities.
        e.g. {(0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.3, (1, 1): 0.7}
    states : dict[str, list]  (optional)
        Maps variable name → list of its possible state values.
        If None, binary {0, 1} is assumed for all variables.
    """

    def __init__(self, variables, table, states=None):
        self.variables = list(variables)
        self.table     = dict(table)
        # States: default binary if not provided
        if states is None:
            self.states = {v: [0, 1] for v in self.variables}
        else:
            self.states = dict(states)

    # ------------------------------------------------------------------
    def _iter_assignments(self, var_list):
        """Iterate over all assignments for var_list respecting their states."""
        return itertools.product(*[self.states.get(v, [0, 1]) for v in var_list])

    # ------------------------------------------------------------------
    def condition_on_evidence(self, evidence: dict) -> "Factor":
        """
        Restrict the factor to the rows consistent with observed evidence.
        Evidence values must match the state representation used in the table.
        """
        new_variables = [v for v in self.variables if v not in evidence]
        idx_keep = [i for i, v in enumerate(self.variables) if v not in evidence]
        idx_ev   = [(i, evidence[v]) for i, v in enumerate(self.variables) if v in evidence]

        new_table = {}
        for assignment, val in self.table.items():
            match = all(assignment[i] == ev_val for i, ev_val in idx_ev)
            if match:
                new_key = tuple(assignment[i] for i in idx_keep)
                new_table[new_key] = val

        new_states = {v: self.states[v] for v in new_variables}
        return Factor(new_variables, new_table, new_states)

    # ------------------------------------------------------------------
    def multiply(self, other: "Factor") -> "Factor":
        """
        Pointwise product of two factors over their combined variable set.
        """
        combined_vars = list(self.variables)
        for v in other.variables:
            if v not in combined_vars:
                combined_vars.append(v)

        # Merge state dictionaries
        combined_states = {v: self.states.get(v, other.states.get(v, [0, 1]))
                           for v in combined_vars}

        self_idx  = [combined_vars.index(v) for v in self.variables]
        other_idx = [combined_vars.index(v) for v in other.variables]

        new_table = {}
        for tuple_vals in itertools.product(
            *[combined_states[v] for v in combined_vars]
        ):
            self_key  = tuple(tuple_vals[i] for i in self_idx)
            other_key = tuple(tuple_vals[i] for i in other_idx)
            v1 = self.table.get(self_key,  0.0)
            v2 = other.table.get(other_key, 0.0)
            new_table[tuple_vals] = v1 * v2

        return Factor(combined_vars, new_table, combined_states)

    # ------------------------------------------------------------------
    def marginalize(self, var: str) -> "Factor":
        """Sum out variable `var` from this factor."""
        if var not in self.variables:
            return self

        var_idx  = self.variables.index(var)
        new_vars = [v for v in self.variables if v != var]
        new_table = {}

        for assignment, val in self.table.items():
            reduced_key = tuple(
                assignment[i] for i in range(len(assignment)) if i != var_idx
            )
            new_table[reduced_key] = new_table.get(reduced_key, 0.0) + val

        new_states = {v: self.states[v] for v in new_vars}
        return Factor(new_vars, new_table, new_states)

    # ------------------------------------------------------------------
    def normalize(self) -> "Factor":
        """Normalize so entries sum to 1.0."""
        total = sum(self.table.values())
        if total > 1e-15:
            for k in self.table:
                self.table[k] /= total
        return self

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"Factor({self.variables}, {len(self.table)} entries)"


# ===========================================================================
# BayesianNetwork
# ===========================================================================

class BayesianNetwork:
    """
    DAG representation of the Bayesian Network with support for multi-state
    nodes (e.g. RiskLevel with states Low/Medium/High).

    Methods
    -------
    add_node(name, parents, states)
        Register a node with its parents and discrete state space.
    is_dag()
        Verify acyclicity via Kahn's topological sort.
    build_factors(params)
        Construct all Factor objects from CPT parameter dict.
    get_ancestors(target_nodes)
        Return all ancestor nodes (for VE pruning).
    get_markov_blanket(node)
        Return Markov blanket (for Gibbs sampling).
    """

    def __init__(self):
        self.nodes    : set  = set()
        self.parents  : dict = {}   # node → list[parent]
        self.children : dict = {}   # node → list[child]
        self.node_states: dict = {} # node → list of states
        self.cpd_specs : dict = {}

    # ------------------------------------------------------------------
    def add_node(self, name: str, parents: list = None, states: list = None):
        """Register a node in the DAG."""
        self.nodes.add(name)
        if parents is None:
            parents = []
        if states is None:
            states = [0, 1]    # default binary
        self.parents[name]      = list(parents)
        self.node_states[name]  = list(states)
        if name not in self.children:
            self.children[name] = []
        for p in parents:
            if p not in self.nodes:
                self.nodes.add(p)
            if p not in self.children:
                self.children[p] = []
            if name not in self.children[p]:
                self.children[p].append(name)

    # ------------------------------------------------------------------
    def set_cpd(self, name: str, factor: "Factor"):
        self.cpd_specs[name] = factor

    # ------------------------------------------------------------------
    def is_dag(self) -> bool:
        """
        Verify acyclicity using Kahn's algorithm (topological sort).
        Returns True if the graph is a valid DAG.
        """
        in_degree = {n: len(self.parents.get(n, [])) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in self.children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return visited == len(self.nodes)

    # ------------------------------------------------------------------
    def get_ancestors(self, target_nodes) -> set:
        """Return the set of all ancestors of the given target nodes (inclusive)."""
        ancestors = set(target_nodes)
        queue = list(target_nodes)
        while queue:
            curr = queue.pop(0)
            for p in self.parents.get(curr, []):
                if p not in ancestors:
                    ancestors.add(p)
                    queue.append(p)
        return ancestors

    # ------------------------------------------------------------------
    def get_markov_blanket(self, node: str) -> set:
        """
        Markov blanket = Parents ∪ Children ∪ Co-parents of children.
        Used in Gibbs sampling to compute the conditional of each node.
        """
        mb = set(self.parents.get(node, []))
        for ch in self.children.get(node, []):
            mb.add(ch)
            for coparent in self.parents.get(ch, []):
                if coparent != node:
                    mb.add(coparent)
        return mb

    # ------------------------------------------------------------------
    def topological_order(self) -> list:
        """Return nodes in topological order (parents before children)."""
        in_deg = {n: len(self.parents.get(n, [])) for n in self.nodes}
        queue  = [n for n in self.nodes if in_deg[n] == 0]
        order  = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children.get(node, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return order

    # ------------------------------------------------------------------
    def build_factors(self, params: dict) -> list:
        """
        Construct all Factor objects from the CPT parameter dictionary
        produced by CERTCPTBuilder.

        params structure:
        {
          "nodes": {
            node_name: {
              "parents": [...],
              "states":  [...],          # this node's states
              "cpt": {                   # explicit CPT
                (parent_states...): {child_state: prob, ...}
              },
              OR
              "noisy_or": {             # Noisy-OR specification (binary nodes only)
                "weights": [...],
                "leak":    float
              }
            }
          }
        }
        """
        factors = []

        for node_name, spec in params["nodes"].items():
            parent_names = spec.get("parents", [])
            node_st      = spec.get("states", [0, 1])

            # Register in DAG
            self.add_node(node_name, parents=parent_names, states=node_st)

            parent_states = [
                params["nodes"][p]["states"] if p in params["nodes"]
                else self.node_states.get(p, [0, 1])
                for p in parent_names
            ]

            # Build all-variable state dict for the Factor
            all_vars   = parent_names + [node_name]
            all_states = {
                v: (params["nodes"][v]["states"]
                    if v in params["nodes"]
                    else self.node_states.get(v, [0, 1]))
                for v in all_vars
            }

            table = {}

            if "cpt" in spec:
                # Explicit CPT: spec["cpt"][(parent_assignment)] = {state: prob}
                cpt = spec["cpt"]
                for p_assign in itertools.product(*parent_states):
                    dist = cpt.get(p_assign, {})
                    for st in node_st:
                        full_key = p_assign + (st,)
                        table[full_key] = float(dist.get(st, 0.0))

            elif "noisy_or" in spec:
                # Noisy-OR (binary 0/1 only)
                weights = spec["noisy_or"]["weights"]
                leak    = spec["noisy_or"]["leak"]
                for p_assign in itertools.product(*parent_states):
                    p_active = _noisy_or(p_assign, weights, leak)
                    table[p_assign + (1,)] = p_active
                    table[p_assign + (0,)] = 1.0 - p_active

            elif "prior" in spec:
                # Root node with prior distribution {state: prob}
                dist = spec["prior"]
                for st in node_st:
                    table[(st,)] = float(dist.get(st, 0.0))
                all_vars   = [node_name]
                all_states = {node_name: node_st}

            else:
                raise ValueError(f"Node '{node_name}' has no 'cpt', 'noisy_or', or 'prior' spec.")

            f = Factor(all_vars, table, all_states)
            self.set_cpd(node_name, f)
            factors.append(f)

        return factors

    # ------------------------------------------------------------------
    def summary(self):
        """Print a brief human-readable summary of the network."""
        order = self.topological_order()
        lines = [
            "=" * 60,
            f"  Bayesian Network  ({len(self.nodes)} nodes)",
            f"  DAG valid: {self.is_dag()}",
            "=" * 60,
        ]
        for node in order:
            parents = self.parents.get(node, [])
            states  = self.node_states.get(node, [0, 1])
            parent_str = ", ".join(parents) if parents else "—"
            lines.append(f"  {node:<28}  states={states}  parents=[{parent_str}]")
        print("\n".join(lines))


# ===========================================================================
# Noisy-OR helper
# ===========================================================================

def _noisy_or(parent_values: tuple, weights: list, leak: float) -> float:
    """
    P(node = 1 | parent_values) under Noisy-OR:
    P = 1 - (1 - leak) × Π_{i: parent_i=1} (1 - w_i)
    """
    prod = 1.0 - leak
    for val, w in zip(parent_values, weights):
        if val == 1:
            prod *= (1.0 - w)
    return 1.0 - prod
