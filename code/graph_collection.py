import argparse
import secrets
from typing import List, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm import tqdm


class GraphCollection:
    """
    A collection of graphs that can grow through recombination operations.

    This class encapsulates a set of graphs and provides methods for growing
    them through random recombination, analyzing their properties, and visualizing
    the results.

    Attributes:
        gset (List[ig.Graph]): List of graph objects in the collection
        gsizes (np.ndarray): Array tracking the size of each graph
        gassembly (np.ndarray): Array tracking the assembly index of each graph
        rng (np.random.Generator): Random number generator for reproducibility
    """

    def __init__(self, N: int, seed=None) -> None:
        """
        Initialize a GraphCollection with N isolated graphs.

        Args:
            N (int): Number of initial graphs to create
            seed (int, optional): Random seed for reproducibility.
                                If None, uses default RNG.

        Returns:
            None
        """
        rng = np.random.default_rng(seed) if seed else np.random.default_rng(42)
        self.gset, self.gsizes, self.gassembly = self._initialize(N, rng)

    def _initialize(
        self, N: int, rng: np.random.Generator
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Initialize a set of N isolated graphs, each with a single node.

        This is a private helper method that sets up the initial state of the
        graph collection.

        Args:
            N (int): Number of graphs to initialize
            rng (np.random.Generator): Random number generator

        Returns:
            Tuple[List, np.ndarray, np.ndarray]: A tuple containing:
                - gset: List of igraph Graph objects (each with 1 node)
                - gsizes: Array of ones (size 1 for each graph)
                - gassembly: Array of zeros (initial assembly index)
        """
        self.gset = [None] * N
        self.gsizes = np.ones(N)
        self.gassembly = np.zeros(N)
        self.rng = rng
        for i in range(N):
            self.gset[i] = ig.Graph(1)
        return self.gset, self.gsizes, self.gassembly

    def grow(self, max_degree: int, times: int) -> None:
        """
        Grow the graph collection through random recombination.

        This method performs 'times' recombination operations where graphs are
        randomly connected or cycles are formed within graphs.

        Args:
            max_degree (int): Maximum allowed degree for any node
            times (int): Number of recombination operations to perform

        Returns:
            None (modifies the collection in place)

        Note:
            The recombination process follows specific rules:
            - Single-node graphs can combine to form edges
            - Larger graphs can connect through random node pairs
            - Cycles can form within individual graphs
            - No self-loops are created
        """
        self.gset, self.gsizes, self.gassembly = self._recombine(
            self.gset, self.gsizes, self.gassembly, max_degree, times, self.rng
        )

    def _recombine(
        self,
        gset: List,
        gsizes: np.ndarray,
        gassembly: np.ndarray,
        max_degree: int,
        times: int,
        rng: np.random.Generator,
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Internal recombination algorithm (private method).

        This method implements the core graph recombination logic where:
        1. Random graph pairs are selected
        2. Nodes are connected between graphs or within graphs
        3. Graph sizes and assembly indices are updated

        Args:
            gset (List[ig.Graph]): List of graphs to recombine
            gsizes (np.ndarray): Current sizes of each graph
            gassembly (np.ndarray): Current assembly indices
            max_degree (int): Maximum node degree allowed
            times (int): Number of recombination operations
            rng (np.random.Generator): Random number generator

        Returns:
            Tuple[List, np.ndarray, np.ndarray]: Updated (gset, gsizes, gassembly)

        Note:
            This is a private method that modifies graphs in place and should
            only be called through the public grow() method.
        """
        # Preallocate the random numbers to avoid multiple calls
        N = len(gset)
        rands = rng.integers(N, size=[times, 2])
        for i in tqdm(range(times), desc="Combining Graphs"):
            rand1 = rands[i, 0]
            rand2 = rands[i, 1]
            obj1 = gset[rand1]
            obj2 = gset[rand2]
            size1 = obj1.vcount()
            size2 = obj2.vcount()
            # If both elements have only one vertex
            # the second one gets a new node.
            if size1 == size2 == 1:
                obj1.add_vertex()
                obj1.add_edge(0, 1)
                gsizes[rand1] = 2
                gassembly[rand1] = 1
            else:
                # Select a random node on each component
                node1 = rng.integers(size1)
                node2 = node1

                while node1 == node2:  # avoids self-loops
                    node2 = rng.integers(size1 + size2)  # allowing cycles to form
                if obj1.vs[node1].degree() < max_degree:
                    # If the selected nodes are from different objects
                    if node2 >= size1:
                        # Create a graph with two connected components
                        obj3 = obj1.disjoint_union(obj2)
                        # Link the two disjoint networks node1-node2
                        obj3.add_edge(node1, node2)

                        # Replace the first element with the new graph
                        gset[rand1] = obj3
                        gsizes[rand1] += gsizes[rand2]
                        gassembly[rand1] = max(gassembly[rand1], gassembly[rand2]) + 1
                    # If the nodes are from the same object -> cycle
                    else:
                        edges = [(edge.source, edge.target) for edge in obj1.es()]
                        if ((node1, node2) in edges) or ((node2, node1) in edges):
                            continue
                        else:
                            obj1.add_edge(node1, node2)
                            gset[rand1] = obj1
        return gset, gsizes, gassembly

    def get_metrics(self) -> pd.DataFrame:
        """
        Calculate and return metrics for all graphs in the collection.

        Computes various graph-theoretic metrics for each graph in the collection
        and returns them as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing metrics for each graph with columns:
                - sizes: Number of nodes in each graph
                - cyclomatic_cpxt: Cyclomatic complexity (Edges - Nodes + 2)
                - min_cycle_length: Length of minimum cycle (0 if acyclic)
                - diameter: Graph diameter (longest shortest path)
                - max_assembly: Assembly index upper bound

        Note:
            Cyclomatic complexity measures the complexity of the graph structure.
            Minimum cycle length is derived from minimum node degree.
            Diameter calculation assumes connected graphs.
            Returns 1 if the graph has a Hamiltonian Cycle, 0 otherwise.
        """
        return self._metrics(self.gset, self.gsizes, self.gassembly)

    def _metrics(
        self, gset: List, gsizes: np.ndarray, gassembly: np.ndarray
    ) -> pd.DataFrame:
        """
        Internal metrics calculation (private method).

        Computes graph metrics for analysis and comparison.

        Args:
            gset (List[ig.Graph]): List of graph objects
            gsizes (np.ndarray): Array of graph sizes
            gassembly (np.ndarray): Array of assembly indices

        Returns:
            pd.DataFrame: DataFrame with computed metrics

        Note:
            This private method contains the core metric calculations and should
            only be accessed through the public get_metrics() method.
        """
        set_size = len(gset)
        cyclomatic_cpxt = np.zeros(set_size)
        min_cycle_length = np.zeros(set_size)
        diameter = np.zeros(set_size)
        has_hamc = np.zeros(set_size)
        for i in tqdm(range(set_size), desc="Analyzing result"):
            component = gset[i]
            cyclomatic_cpxt[i] = (
                component.ecount() - component.vcount() + 2
            )  # only 1 connected component by definition
            min_deg = min(component.degree())  # Preposition 2.11.1 Graph Theory notes
            if min_deg > 1:
                min_cycle_length[i] = min_deg + 1
            diameter[i] = component.diameter()
            # Applying Dirac's Criterion (Graph Theory Notes Corollary 2.14.5)
            has_hamc[i] = 0 if min(gset[i].degree()) < set_size / 2 else 1

        metrics = pd.DataFrame(
            {
                "sizes": gsizes,
                "cyclomatic_cpxt": cyclomatic_cpxt,
                "min_cycle_length": min_cycle_length,
                "diameter": diameter,
                "max_assembly": gassembly,
                "has_hamc": has_hamc,
            }
        )
        return metrics

    def represent(self):
        """
        Visualize the entire graph collection.

        Creates a compound graph from all graphs in the collection and
        displays it with colored components.

        Returns:
            None (displays matplotlib plot)

        Note:
            For large collections, this may produce a complex visualization.
            Components are colored differently for clarity.
        """
        multi_graph = self._join_graphs(self.gset)
        self._represent(multi_graph)

    def _represent(self, g: ig.Graph) -> None:
        """
        Internal visualization method (private).

        Renders a graph with colored components using igraph and matplotlib.

        Args:
            g (ig.Graph): Graph to visualize

        Returns:
            None (displays the plot)

        Note:
            Uses rainbow palette for component coloring and adjusts
            vertex/edge sizes for better visualization.
        """
        components = g.connected_components(mode="weak")
        fig, ax = plt.subplots()
        ig.plot(
            components,
            target=ax,
            palette=ig.RainbowPalette(),
            vertex_size=7,
            vertex_color=list(
                map(int, ig.rescale(components.membership, (0, 200), clamp=True))
            ),
            edge_width=0.7,
        )
        plt.show()

    def _join_graphs(self, gset: List):
        """
        Create compound graph from collection (private method).

        Combines multiple graphs into one using disjoint union operations.

        Args:
            gset (List[ig.Graph]): List of graphs to join

        Returns:
            ig.Graph: Compound graph containing all input graphs as disjoint components

        Note:
            This preserves the original graphs as separate components in
            the compound graph, useful for visualization and analysis.
        """
        compound = ig.Graph()
        for g in gset:
            compound = compound.disjoint_union(g)
        return compound


def main(
    N: int,
    max_degree: int,
    time_steps: int,
    rng: np.random.Generator,
    graph: bool = False,
) -> GraphCollection:
    """
    Main function to run the graph recombination simulation.

    Orchestrates the complete workflow:
    1. Creates a graph collection
    2. Grows it through recombination
    3. Computes and displays metrics
    4. Optionally visualizes the result

    Args:
        N (int): Number of initial graphs in the collection
        max_degree (int): Maximum allowed degree for any node
        time_steps (int): Number of recombination operations to perform
        rng (np.random.Generator): Random number generator for reproducibility
        graph (bool): Whether to display graph visualization (default: False)

    Returns:
        GraphCollection: The grown graph collection object

    Note:
        This function displays several plots:
        - Histogram of graph sizes
        - Scatter plot of cyclomatic complexity vs diameter
        - Optional graph visualization if graph=True
    """
    collection = GraphCollection(N, rng)
    collection.grow(max_degree, time_steps)
    data = collection.get_metrics()

    max_index = np.argmax(collection.gsizes)
    max_size = collection.gsizes[max_index]
    max_assembly = collection.gassembly[max_index]
    print("Biggest element size: ", max_size)
    print("Assembly index upper bound: ", max_assembly)

    sns.histplot(data.sizes)
    plt.yscale("log")
    plt.show()

    scatter = sns.scatterplot(
        data,
        x="diameter",
        y="cyclomatic_cpxt",
        c=data.has_hamc,
        label="Generated data",
    )
    plt.xlabel("Component Diameter")
    plt.ylabel("Cyclomatic Complexity")
    plt.title("Cyclomatic Complexity vs Diameter per component")
    plt.colorbar(scatter.collections[0], label="Has hamc")
    plt.legend()
    plt.grid(True)
    plt.show()

    if graph == True:
        collection.represent()
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="graph_collection",
        usage="%(prog)s set_size max_degree iterations bonds_per_iteration [options]",
        description="Script that generates a collection of graphs from a simple combination rule",
        epilog="Example:  python3 graph_collection.py 100 6 150 -g True -s True",
    )
    parser.add_argument("set_size", type=int, help="Size of the set")
    parser.add_argument(
        "max_degree", type=int, help="Max number of edges for each node"
    )
    parser.add_argument(
        "iterations", type=int, help="Iterations where 'b' new bonds are created"
    )
    parser.add_argument(
        "-g",
        type=bool,
        help="Toggle for showing a representation of the set. Recommended only for smaller (n<1000) sets",
    )
    parser.add_argument("-s", type=bool, help="Use a hardcoded seed")
    args = parser.parse_args()

    semilla = 42 if args.s else secrets.randbits(128)
    rng = np.random.default_rng(semilla)
    main(
        args.set_size,
        args.max_degree,
        args.iterations,
        rng,
        args.g,
    )
