import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def string_entropy(string: str) -> float:
    n = len(string)
    n0 = string.count("0")
    n1 = n - n0
    p0 = n0 / n
    p1 = n1 / n
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def init_pool(
    n_0: int, n_1: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pool = np.array(["0", "1"])
    copies = np.array([n_0, n_1])
    indexes = np.array([0, 0])
    entropies = np.zeros(2)
    entropies = np.array([string_entropy(string) for string in pool])
    return pool, copies, indexes, entropies


def combine(
    pool: np.ndarray, copies: np.ndarray, indexes: np.ndarray, entropies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.sum(copies)
    items = np.random.choice(pool, size=2, p=copies / n)
    new_copies = copies
    new_indexes = indexes
    new_entropies = entropies

    new_element = items[0] + items[1]
    if new_element in pool:
        new_pool = pool
        idx = np.where(pool == new_element)
        new_copies[idx] += 1
        new_indexes = indexes
    else:
        new_pool = np.append(pool, new_element)
        new_copies = np.append(new_copies, 1)
        new_entropies = np.append(entropies, string_entropy(new_element))
    return new_pool, new_copies, new_entropies


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pool, copies, indexes, entropies = init_pool(1000, 1000)
    n = np.sum(copies)
    probs = copies / n
    for i in tqdm(range(15000), desc="Iterations: "):
        new_pool, new_copies, new_entropies = combine(pool, copies, indexes, entropies)
        pool = new_pool
        copies = new_copies
        entropies = new_entropies

    return pool, copies, indexes, entropies


if __name__ == "__main__":
    pool, copies, indexes, entropies = main()
    sizes = np.array([len(x) for x in pool])
    scatter = sns.scatterplot(x=sizes, y=copies, hue=entropies)
    plt.grid(which="both", linestyle=":")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Size")
    plt.ylabel("Copy number")
    plt.title("Copies of bitstrings by length")
    plt.legend()
    plt.show()

    plt.hist(entropies)
    plt.yscale("log")
    plt.show()
