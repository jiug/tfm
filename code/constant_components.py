import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


def initialize(N):
    gset = {}
    for i in range(N):
        gset[i] = ig.Graph(1)
    return gset


def recombine(gset, times, times_2):
    # Preallocate the random numbers to avoid multiple calls
    rands = np.random.randint(N, size=[times, times_2, 2])
    max_nodes = [0, 2]
    for i in range(times):
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
                obj2.add_vertex()
                obj2.add_edge(0, 1)
            else:
                # Create a graph with two connected components
                obj3 = obj1.disjoint_union(obj2)

                # Select a random node on each component (of degree < D)
                # and link them
                node1 = np.random.randint(size1)
                node2 = size1 + np.random.randint(size2)  # due to indexing
                obj3.add_edge(node1, node2)
                nodes = obj3.vcount()
                if nodes > max_nodes[1]:
                    max_nodes = [rand2, obj3.vcount()]

                gset[rand2] = obj3
                print(max_nodes)
    return gset, max_nodes


def find_largest(gset):
    # Make an array with the nodes of every graph in the gset
    node_count = np.array([gset[graph].vcount() for graph in gset])
    idx = np.max(node_count)
    print(node_count)
    return idx


def represent(g):
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


def main(N, rng):
    gset = initialize(N)
    gset, max_nodes = recombine(gset, 10, 10)
    represent(gset[max_nodes[0]])
    node_count = [gset[graph].vcount() for graph in gset]


if __name__ == "__main__":
    semilla = 51001430439489238069396834186967689176
    rng = np.random.default_rng(semilla)
    N = 100
    main(N, rng)
