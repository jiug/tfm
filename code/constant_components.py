import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
import argparse
import seaborn as sns
import secrets


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
        rng: Random number generator

    Returns:
        A tuple containing the updated gset, gsizes, and gassembly arrays

    Note:
        This function modifies the input graphs in place and updates the tracking arrays.
    """
    # Preallocate the random numbers to avoid multiple calls
    N = len(gset)
    rands = rng.integers(N, size=[times, 2])
    for i in tqdm(range(times), desc="Combining Graphs"):
        rand1 = rands[i,0]
        rand2 = rands[i,1]
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
                    edges = [ (edge.source, edge.target) for edge in obj1.es()]
                    if((node1, node2) in edges) or ((node2,node1) in edges) :
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

def metrics(gset:List, gsizes:np.ndarray, gassembly:np.ndarray) -> pd.DataFrame : 
    """
    Combine measurements of the collection of graphs and into a pd.DataFrame
    Args:
        gset:       List of igraph Graph objects to be joined
        gsize:      List of sizes corresponding to the elements in gset
        gassembly   List of upper limits for the assembly index corresponding
                    to the elements in gset
    Returns:
        metrics:    DataFrame with the following metrics for each element
                        -sizes:             number of nodes  
                        -cyclomatic_cpxt:   cyclomatic complexity = Edges - Nodes + 2 
                        -min_cycle_length:  Length of the minimum cycle (0 if None)
                        -diameter:          Maximum of the list of shortests paths
                        -max_asselbly:      Index that correlates  with the steps needed
                                            to arrive to the given component by recombination

    Note:
        The function returns a pd.DataFrame with all the metrics.
    """
    set_size = len(gset) 
    cyclomatic_cpxt = np.zeros(set_size)
    min_cycle_length = np.zeros(set_size)
    diameter = np.zeros(set_size)
    for i in tqdm(range(set_size),desc='Analyzing result'):
        component = gset[i]
        cyclomatic_cpxt[i] = component.ecount() - component.vcount() + 2 #only 1 connected component by definition
        min_deg = min(component.degree()) #Preposition 2.11.1 Graph Theory notes
        if min_deg > 1:
            min_cycle_length[i] = min_deg + 1
        diameter[i] = component.diameter()

    metrics = pd.DataFrame({'sizes':gsizes, 'cyclomatic_cpxt': cyclomatic_cpxt, 'min_cycle_length': min_cycle_length, 'diameter': diameter, 'max_assembly': gassembly})
    return metrics


def main(
    N: int,
    max_degree: int,
    time_steps: int,
    rng: np.random.Generator,
    graph: bool = False,
)-> tuple[list, np.ndarray, np.ndarray]:
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
        gset, gsizes, gassembly, max_degree, time_steps, rng
    )
    max_index = np.argmax(gsizes)
    max_size = gsizes[max_index]
    max_assembly = gassembly[max_index]
    print("Biggest element size: ", max_size)
    print("Assembly index upper bound: ", max_assembly)

    data = metrics(gset, gsizes, gassembly)
   
    sns.histplot(data.sizes)
    plt.show()

    scatter = sns.scatterplot(data, x= 'diameter',y='cyclomatic_cpxt',c=data.max_assembly)
    plt.xlabel('Component Diameter')
    plt.ylabel('Cyclomatic Complexity')
    plt.title('Cyclomatic Complexity vs Diameter per component')
    plt.colorbar(scatter.collections[0], label='Graph size (log)')
    plt.show()

    if graph == True:
        compound = join_graphs(gset)
        represent(compound)

    return gset, gsizes, gassembly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="constant_components",
        usage="%(prog)s set_size max_degree iterations bonds_per_iteration [options]",
        description="Script that generates a collection of graphs from a simple combination rule",
        epilog="Example:  python3 constant_components.py 100 6 150 -g True -s True",
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

    if args.s:
        semilla = 51001430439489238069396834186967689176
    else:
        semilla = secrets.randbits(128)
    rng = np.random.default_rng(semilla)
    main(
        args.set_size,
        args.max_degree,
        args.iterations,
        rng,
        args.g,
    )
