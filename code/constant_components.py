import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


def initialize(N):
    gset = {}
    gsizes = np.ones(N)
    gassembly = np.zeros(N)
    for i in range(N):
        gset[i] = ig.Graph(1)
    return gset, gsizes


def recombine(gset, gsizes, times, times_2, rng):
    # Preallocate the random numbers to avoid multiple calls
    rands = rng.integers(N, size=[times, times_2, 2])
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
                obj1.add_vertex()
                obj1.add_edge(0, 1)
                gsizes[rand1] = 2
            else:
                # Select a random node on each component
                # TODO: limit the maximum node degree
                node1 = rng.integers(size1)
                node2 = node1
                while node1 == node2:  # avoids self-loops
                    node2 = rng.integers(size1 + size2)  # allowing cycles to form

                # In case there isn't a cycle
                if node2 >= size1:
                    # Create a graph with two connected components
                    obj3 = obj1.disjoint_union(obj2)
                    # Link the two disjoint networks node1-node2
                    obj3.add_edge(node1, node2)
                    nodes = obj3.vcount()

                    # Replace the first element with the new graph
                    gset[rand1] = obj3
                    gsizes[rand1] += gsizes[rand2]
                else:
                    if (node1, node2) in obj1.es():
                        continue
                    else:
                        obj1.add_edge(node1, node2)
                        gset[rand1] = obj1
    return gset, gsizes


def find_largest(gset):
    # Make an array with the nodes of every graph in the gset
    node_count = np.array([gset[graph].vcount() for graph in gset])
    idx = np.max(node_count)
    # print(node_count)
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
    gset, gsizes = initialize(N)
    gset, gsizes = recombine(gset, gsizes, 100, 15, rng)
    max_index = np.argmax(gsizes)
    max_size = gsizes[max_index]
    print("Largest element index: ", max_index)
    print("Element size: ", max_size)
    represent(gset[max_index])

    # represent(gset[max_nodes[0]])


if __name__ == "__main__":
    semilla = 51001430439489238069396834186967689176
    rng = np.random.default_rng(semilla)
    N = 400
    main(N, rng)
