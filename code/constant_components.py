import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def initialize(N: int) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Initialize a set of N isolated graphs, each with a single node.

    Args:
        N: Number of graphs to initialize.

    Returns:
        A tuple containing:
        - gset: List of igraph Graph objects
        - gsizes: Array tracking the size (number of nodes) of each graph
        - gassembly: Array tracking the assembly index of each graph
    """
    gset = [None] * N
    gsizes = np.ones(N)
    gassembly = np.zeros(N)
    for i in range(N):
        gset[i] = ig.Graph(1)
    return gset, gsizes, gassembly


def recombine(
    gset: List,
    gsizes: np.ndarray,
    gassembly: np.ndarray,
    max_degree: int,
    times: int,
    times_2: int,
    rng: np.random.Generator,
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Recombine graphs by randomly connecting nodes from different graphs or creating cycles.

    Args:
        gset: List of graphs to recombine
        gsizes: Array tracking the size of each graph
        gassembly: Array tracking the assembly index of each graph
        max_degree: Maximum allowed degree for nodes
        times: Generation steps
        times_2: Number of new bonds created per generation step
        rng: Random number generator

    Returns:
        A tuple containing the updated gset, gsizes, and gassembly arrays

    Note:
        This function modifies the input graphs in place and updates the tracking arrays.
    """
    # Preallocate the random numbers to avoid multiple calls
    N = len(gset)
    rands = rng.integers(N, size=[times, times_2, 2])
    for i in tqdm(range(times), desc="Outer loop"):
        for j in range(times_2):
            rand1 = rands[i, j, 0]
            rand2 = rands[i, j, 1]
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
                        if (node1, node2) in obj1.es():
                            continue
                        else:
                            obj1.add_edge(node1, node2)
                            gset[rand1] = obj1
    return gset, gsizes, gassembly


def represent(g: ig.Graph) -> None:
    """
    Visualize a graph with colored components.

    Args:
        g: The graph to visualize

    Returns:
        None (displays the plot)
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


def join_graphs(gset: List):
    """
    Combine multiple graphs into a single compound graph using disjoint unions.

    Args:
        gset: List of igraph Graph objects to be joined

    Returns:
        A single igraph Graph object containing all the input graphs as disjoint components

    Note:
        The function creates a new compound graph by sequentially performing disjoint
        union operations on all graphs in the input list.
    """
    compound = ig.Graph()
    for g in gset:
        compound = compound.disjoint_union(g)
    return compound


def main(
    N: int,
    max_degree: int,
    time_steps: int,
    bonds_per_step: int,
    rng: np.random.Generator,
) -> None:
    """
    Main function to run the graph recombination simulation.

    Args:
        N: Number of initial graphs
        max_degree: Maximum allowed degree for nodes
        rng: Random number generator

    Returns:
        None
    """
    gset, gsizes, gassembly = initialize(N)
    gset, gsizes, gassembly = recombine(
        gset, gsizes, gassembly, max_degree, time_steps, bonds_per_step, rng
    )
    max_index = np.argmax(gsizes)
    max_size = gsizes[max_index]
    max_assembly = gassembly[max_index]
    print("Biggest element size: ", max_size)
    print("Assembly index upper bound: ", max_assembly)
    compound = join_graphs(gset)
    represent(compound)


if __name__ == "__main__":
    semilla = 51001430439489238069396834186967689176
    rng = np.random.default_rng(semilla)
    N = 100
    max_degree = 6
    time_steps = 10
    bonds_per_step = 15
    main(N, max_degree, time_steps, bonds_per_step, rng)
