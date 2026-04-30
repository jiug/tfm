import argparse
import secrets
from collections import defaultdict
from typing import List, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm import tqdm


class GraphPool:
    def __init__(
        self, N: int, seed: int, max_degree: int, batch_size: int = 250
    ) -> None:
        self.rng = (
            np.random.default_rng(seed) if seed else np.random.default_rng(985281467)
        )
        self._batch_size = batch_size
        self._max_degree = max_degree
        self._pending_elements = []
        self.pool, self.evolution = self._initialize(N, self.rng)
        self._signature_to_indices = {}
        self._build_signature_lookup()

    def _initialize(
        self, N: int, rng: np.random.Generator
    ) -> Tuple[pd.DataFrame, dict]:
        """Initialize pool with N single-node graphs."""
        initial_graphs = [ig.Graph(1) for _ in range(N)]

        pool = {
            "element": initial_graphs,
            "copy_number": [1] * N,
            "size": [1] * N,
            "steps": [0] * N,
            "assembly": [0] * N,
            "assembly_efficiency": [1.0] * N,
            "cyclomatic_complexity": [1] * N,
            "signature": [self._graph_signature(g) for g in initial_graphs],
            "degree_entropy": [0.0] * N,
            "is_tree": [True] * N,
            "diameter": [0] * N,
            "clustering_coefficient": [0.0] * N,
            "is_connected": [True] * N,
        }

        history = {
            "pool_size": [N],
            "steps": [0],
            "assembly": [0],
            "average_assembly": [0],
            "cyclomatic_complexity": [1],
            "ensemble_entropy": [0.0],
        }
        return pd.DataFrame(pool), history

    def _build_signature_lookup(self) -> None:
        """Build dictionary mapping signatures to pool indices."""
        self._signature_to_indices = defaultdict(list)
        for idx, sig in enumerate(self.pool.signature):
            self._signature_to_indices[sig].append(idx)

    def _graph_signature(self, graph: ig.Graph) -> tuple:
        """Compute a hashable signature for fuzzy equality."""
        degrees = sorted(graph.degree())
        return (
            graph.vcount(),  # Stage 1: vertex count
            graph.ecount(),  # Stage 2: edge count
            tuple(degrees),  # Stage 3: degree sequence
            self._cyclomatic_complexity(graph),  # Stage 4: cyclomatic complexity
        )

    def _cyclomatic_complexity(self, graph: ig.Graph) -> int:
        """Calculate cyclomatic complexity: ecount - vcount + 2."""
        return graph.ecount() - graph.vcount() + 2

    def _degree_entropy(self, graph: ig.Graph) -> float:
        """Calculate Shannon entropy of degree distribution."""
        degrees = graph.degree()
        if len(degrees) == 0:
            return 0.0
        unique, counts = np.unique(degrees, return_counts=True)
        probs = counts / len(degrees)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _calculate_assembly(self, graph: ig.Graph) -> int:
        """Calculate assembly index for a graph.
        Uses number of spanning trees for connected graphs, edge count otherwise."""
        if graph.is_connected():
            try:
                return int(graph.spanning_tree_count())
            except:
                return graph.ecount()
        return graph.ecount()

    def _update_elements(self) -> None:
        """Add pending elements to the main pool."""
        if len(self._pending_elements) > 0:
            data = pd.DataFrame(self._pending_elements)
            self.pool = pd.concat([self.pool, data], ignore_index=True)
            self._pending_elements = []
            self._build_signature_lookup()

    def _get_valid_nodes(self, graph: ig.Graph) -> List[int]:
        """Get list of nodes with degree < max_degree."""
        if self._max_degree is None:
            return list(range(graph.vcount()))
        return [i for i, deg in enumerate(graph.degree()) if deg < self._max_degree]

    def _combine(self) -> None:
        """Combine two random graphs from the pool using fuzzy equality.

        Uses size-based node selection: nodes are chosen from the combined
        range [0, size1+size2), allowing cycles to form within larger graphs
        and connections between graphs based on their relative sizes.
        """
        non_extinct = self.pool.copy_number > 0

        if np.sum(non_extinct) < 2:
            return

        prob = self.pool.copy_number[non_extinct] / np.sum(
            self.pool.copy_number[non_extinct]
        )
        indices = self.rng.choice(
            np.arange(len(self.pool))[non_extinct], size=2, p=prob, replace=False
        )

        g1, g2 = self.pool.element[indices[0]], self.pool.element[indices[1]]
        size1, size2 = g1.vcount(), g2.vcount()
        total_size = size1 + size2

        # Special case for max_degree=2: always form linear chains
        if self._max_degree == 2:
            valid_nodes_g1 = [i for i, deg in enumerate(g1.degree()) if deg < 2]
            valid_nodes_g2 = [i for i, deg in enumerate(g2.degree()) if deg < 2]

            if not valid_nodes_g1 or not valid_nodes_g2:
                return

            v1 = self.rng.choice(valid_nodes_g1)
            v2 = self.rng.choice(valid_nodes_g2)
            new_graph = g1.disjoint_union(g2)
            v2_adjusted = v2 + size1
            new_graph.add_edge(v1, v2_adjusted)
        else:
            # General case: select nodes from combined range to allow cycles
            # Generate two random node indices in the combined range [0, total_size)
            node1 = self.rng.integers(total_size)
            node2 = self.rng.integers(total_size)
            while node1 == node2:  # Avoid self-loops
                node2 = self.rng.integers(total_size)

            # Check if both nodes are in the same original graph
            if node1 < size1 and node2 < size1:
                # Both in g1 -> create cycle within g1 only
                # Discard the joint graph, modify g1 directly
                if not g1.are_adjacent(node1, node2):
                    if (
                        self._max_degree is None or g1.degree(node1) < self._max_degree
                    ) and (
                        self._max_degree is None or g1.degree(node2) < self._max_degree
                    ):
                        new_graph = g1.copy()
                        new_graph.add_edge(node1, node2)
                    else:
                        return
                else:
                    return
            elif node1 >= size1 and node2 >= size1:
                # Both in g2 -> create cycle within g2 only
                # Discard the joint graph, modify g2 directly
                node1_g2 = node1 - size1
                node2_g2 = node2 - size1
                if not g2.are_adjacent(node1_g2, node2_g2):
                    if (
                        self._max_degree is None
                        or g2.degree(node1_g2) < self._max_degree
                    ) and (
                        self._max_degree is None
                        or g2.degree(node2_g2) < self._max_degree
                    ):
                        new_graph = g2.copy()
                        new_graph.add_edge(node1_g2, node2_g2)
                    else:
                        return
                else:
                    return
            else:
                # One in each -> connect between g1 and g2
                if node1 >= size1:
                    node1, node2 = node2, node1  # Ensure node1 is in g1, node2 in g2
                # node1 is in [0, size1), node2 is in [size1, total_size)
                node2_g2 = node2 - size1
                if (
                    self._max_degree is None or g1.degree(node1) < self._max_degree
                ) and (
                    self._max_degree is None or g2.degree(node2_g2) < self._max_degree
                ):
                    new_graph = g1.disjoint_union(g2)
                    new_graph.add_edge(node1, node2)
                else:
                    return

        # Compute new metrics
        new_steps = max(self.pool.steps[indices]) + 1
        new_size = new_graph.vcount()
        new_signature = self._graph_signature(new_graph)
        new_assembly = self._calculate_assembly(new_graph)

        # Check for fuzzy match
        if new_signature in self._signature_to_indices:
            # For fuzzy equality: increment copy number of first match
            match_idx = self._signature_to_indices[new_signature][0]
            self.pool.at[match_idx, "copy_number"] += 1

            # Update steps if this path was longer
            if new_steps > self.pool.steps[match_idx]:
                self.pool.at[match_idx, "steps"] = new_steps
                self.pool.at[match_idx, "assembly_efficiency"] = (
                    new_steps / new_assembly
                )
        else:
            # Add new element to pending
            new_element = {
                "element": new_graph,
                "copy_number": 1,
                "size": new_size,
                "steps": new_steps,
                "assembly": new_assembly,
                "assembly_efficiency": (
                    new_steps / new_assembly if new_assembly > 0 else 1.0
                ),
                "cyclomatic_complexity": self._cyclomatic_complexity(new_graph),
                "signature": new_signature,
                "degree_entropy": self._degree_entropy(new_graph),
                "is_tree": new_graph.is_tree(),
                "diameter": new_graph.diameter() if new_size > 1 else 0,
                "clustering_coefficient": new_graph.transitivity_undirected(),
                "is_connected": new_graph.is_connected(),
            }
            self._pending_elements.append(new_element)

            if len(self._pending_elements) >= self._batch_size:
                self._update_elements()

    def evolve(self, steps: int) -> pd.DataFrame:
        """Run evolution for given steps, tracking metrics."""
        init_data = [
            {
                "pool_size": len(self.pool),
                "steps": 0,
                "assembly": 0,
                "average_assembly": 0,
                "cyclomatic_complexity": 1,
                "ensemble_entropy": 0.0,
            }
        ]

        for step in tqdm(range(steps), desc="Evolving graph pool"):
            self._combine()

            # Periodically record evolution metrics
            if np.log2(max(1, step)) % 1 == 0:
                non_extinct = self.pool.copy_number > 0
                evol_metrics = {
                    "pool_size": len(self.pool[non_extinct]),
                    "steps": np.max(self.pool.steps[non_extinct]),
                    "assembly": np.max(self.pool.assembly[non_extinct]),
                    "average_assembly": np.mean(self.pool.assembly[non_extinct]),
                    "cyclomatic_complexity": np.max(
                        self.pool.cyclomatic_complexity[non_extinct]
                    ),
                    "ensemble_entropy": self.ensemble_entropy(),
                }
                init_data.append(evol_metrics)

        # Add any remaining pending elements
        self._update_elements()

        evolution = pd.DataFrame(init_data)
        self.evolution = evolution
        return evolution

    def ensemble(self) -> None:
        """Calculate ensemble probabilities."""
        non_extinct = self.pool.copy_number > 0
        n = np.sum(self.pool.copy_number[non_extinct])
        if n > 0:
            self.pool["ensemble"] = self.pool.copy_number / n
        else:
            self.pool["ensemble"] = 0.0

    def ensemble_entropy(self) -> float:
        """Calculate diversity of the graph pool."""
        self.ensemble()
        non_extinct = self.pool.copy_number > 0
        ensemble = self.pool.ensemble[non_extinct]
        return -np.sum(ensemble * np.log2(ensemble + 1e-10))

    def represent(self, max_components: int = 50) -> None:
        """Visualize a sample of graphs from the pool."""
        non_extinct = self.pool.copy_number > 0
        if np.sum(non_extinct) == 0:
            print("No non-extinct graphs to display")
            return

        # Select up to max_components graphs to display
        indices = np.where(non_extinct)[0]
        if len(indices) > max_components:
            indices = self.rng.choice(indices, size=max_components, replace=False)

        # Create compound graph
        compound = ig.Graph()
        for idx in indices:
            compound = compound.disjoint_union(self.pool.element[idx])

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        components = compound.connected_components(mode="weak")
        ig.plot(
            components,
            target=ax,
            palette=ig.RainbowPalette(),
            vertex_size=10,
            vertex_color=list(
                map(int, ig.rescale(components.membership, (0, 200), clamp=True))
            ),
            edge_width=0.7,
        )
        plt.title(f"Sample of {len(indices)} graphs from pool")
        plt.show()


def evol_graph(evolution: pd.DataFrame) -> None:
    """Plot evolution metrics over time."""
    x = evolution.index
    for col in [
        "ensemble_entropy",
        "assembly",
        "average_assembly",
        "cyclomatic_complexity",
        "pool_size",
    ]:
        sns.lineplot(data=evolution, x=x, y=col, label=str(col))
    plt.yscale("log")
    plt.xlabel(r"Log time $s = \log_2(t)$")
    plt.title("Graph Pool Evolution Metrics")
    plt.grid(which="both")
    plt.tight_layout()
    plt.legend()
    plt.show()


def main(
    N: int, steps: int, max_degree: int, fixed_seed: bool = True, graph: bool = False
):
    """Main function to run graph pool simulation."""
    if fixed_seed:
        seed = 916109126
    else:
        seed = secrets.randbits(128)

    gp = GraphPool(N, seed, max_degree)
    evolution = gp.evolve(steps)

    # Plot evolution metrics
    evol_graph(evolution)

    # Plot pool metrics
    sns.pairplot(
        gp.pool[gp.pool.copy_number > 0],
        vars=[
            "copy_number",
            "size",
            "steps",
            "assembly",
            "cyclomatic_complexity",
            "degree_entropy",
        ],
        hue="is_tree",
        diag_kind="hist",
        plot_kws=dict(marker=".", size=2),
    )
    plt.suptitle("Graph Pool Metrics", y=1.02)
    plt.show()

    if graph:
        gp.represent()

    return gp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="graph_pool",
        usage="%(prog)s initial_graphs evolution_steps [options]",
        description="Script that evolves a pool of graphs following simple stochastic combination rules",
        epilog="Example: python graph_pool.py 10 1000 -d 2 -g",
    )
    parser.add_argument("N", type=int, help="Number of initial single-node graphs")
    parser.add_argument("steps", type=int, help="Number of evolution steps")
    parser.add_argument(
        "-d",
        "--max_degree",
        type=int,
        help="Maximum degree for any node (default: no limit)",
        default=None,
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="Show graph visualization"
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="Random seed for reproducibility", default=None
    )
    args = parser.parse_args()

    main(args.N, args.steps, args.max_degree, args.seed, args.graph)
