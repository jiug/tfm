import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def initialize_system(n0: int, t: int, rng, p, new_links):
    G = nx.Graph()
    G.add_nodes_from(range(n0))
    for i in range(t):
        r = rng.uniform(0, 1)

        # This block selects n = newlinks random nodes and adds
        # an edge between them.
        for j in range(new_links):
            if r >= p:
                n_elem = nx.number_connected_components(G)
                u = int(rng.uniform(0, n_elem))
                v = int(rng.uniform(0, n_elem))
                while u == v:
                    v = int(rng.uniform(0, n_elem))
                G.add_edge(u, v)
    return G


def main(n0, t, rng, p, new_links):
    G = initialize_system(n0, t, rng, p, new_links)

    # Histogram of node degrees in graph G
    print(len(G.edges()))
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=range(max(degrees) + 2))
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    # Initialization parameters
    semilla = 51001430439489238069396834186967689176
    rng = np.random.default_rng(semilla)
    n0 = 1000
    t = 3000
    new_links = 4
    p = 0.5
    main(n0, t, rng, p, new_links)
