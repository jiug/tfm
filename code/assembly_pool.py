import math as m

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from oeis_sequences import OEISsequences as oeis
from tqdm import tqdm


class AssemblyPool:
    def __init__(self, n1: int, n0: int) -> None:
        self.pool = self._initialize(n1, n0)

    def _initialize(self, n1, n0) -> pd.DataFrame:
        initial_pool = {
            "element": np.array(["1", "0"]),
            "copy_number": np.array([n1, n0]),
            "entropy": np.array([0, 0]),
            "assembly": np.array([0, 0]),
            "history": np.array(["1", "0"]),
            "steps": np.array([0, 0]),
        }
        pool = pd.DataFrame(data=initial_pool)
        return pool
    
    # Function that concatenates two random elements from the assembly pool
    def combine(self) -> None:
        n = np.sum(self.pool.copy_number)
        prob = self.pool.copy_number / n
        index = np.random.choice(np.arange(len(self.pool.element)), size=2, p=prob)

        element0, element1 = self.pool.element[index]

        # Logig to keep the biggest element always first
        reorg = False
        if len(element0) >= len(element1):
            new_element = element0 + element1
        else:
            new_element = element1 + element0
            reorg = True

        # Fin the new element in the assembly pool
        idx = self.pool.loc[self.pool.element == new_element].index

        # If the new element is not idx will be an empty array
        if idx.size > 0:
            cnum = self.pool.at[idx[0], "copy_number"] + 1
            # Increase the copy number of the existing element
            self.pool.at[idx[0], "copy_number"] = cnum
            return
        else:

            # Bigger elements are always first
            if reorg:
                history = [
                    f"({self.pool.history[index[1]]})({self.pool.history[index[0]]})"
                ]
            else:
                history = [
                    f"({self.pool.history[index[0]]})({self.pool.history[index[1]]})"
                ]
            # Add new element to the assembly pool
            # Same structure as the __init__ dictionary
            new_observation = {
                "element": np.array([new_element]),
                "copy_number": np.array([1]),
                "entropy": np.array([string_entropy(new_element)]),
                "assembly": np.array([0]),
                "history": np.array(history),
                "steps": np.array(np.max(self.pool.steps[index]) + 1),
            }
            # Convert the dictionary to DataFrame
            data = pd.DataFrame(data=new_observation)
            # Add a new observation to the dataframe
            self.pool = pd.concat([self.pool, data], ignore_index=True)
        return

    def interpret(self):
        for i in element:
            if i == '0':
                

def string_entropy(string: str) -> float:
    n = len(string)
    n0 = string.count("0")
    n1 = n - n0
    p0 = n0 / n
    p1 = n1 / n
    if n0 == 0 or n1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


# def init_pool(
#     n_0: int, n_1: int
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     pool = np.array(["0", "1"])
#     copies = np.array([n_0, n_1])
#     indexes = np.array([0, 0])
#     entropies = np.zeros(2)
#     entropies = np.array([string_entropy(string) for string in pool])
#     return pool, copies, indexes, entropies


# def combine(
#     pool: np.ndarray, copies: np.ndarray, indexes: np.ndarray, entropies: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     n = np.sum(copies)
#     items = np.random.choice(pool, size=2, p=copies / n)
#     new_copies = copies
#     new_indexes = indexes
#     new_entropies = entropies
#
#     new_element = items[0] + items[1]
#     if new_element in pool:
#         new_pool = pool
#         idx = np.where(pool == new_element)
#         new_copies[idx] += 1
#         new_indexes = indexes
#     else:
#         new_pool = np.append(pool, new_element)
#         new_copies = np.append(new_copies, 1)
#         new_entropies = np.append(entropies, string_entropy(new_element))
#     return new_pool, new_copies, new_entropies
#


def main(n0, n1) -> None:
    ap = AssemblyPool(n0, n1)
    for i in tqdm(range(10000), desc="Calculating iterations:"):
        ap.combine()

    ap.pool["size"] = ap.pool.element.str.len()
    ap.pool["balanced"] = ap.pool.element.str.count("0") == ap.pool.element.str.count(
        "1"
    )
    sns.scatterplot(data=ap.pool, x="size", y="copy_number", hue="entropy")
    plt.grid(which="both", linestyle=":")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Size")
    plt.ylabel("Copy number")
    plt.title("Copies of bitstrings by length")
    plt.legend()
    plt.show()

    ap.pool["log_copy"] = np.log1p(ap.pool["copy_number"])
    sns.scatterplot(data=ap.pool, x="size", y="entropy", hue="log_copy")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both")
    plt.show()

    sns.histplot(ap.pool.entropy, bins=20)
    plt.yscale("log")
    plt.xlabel("Entropy")
    plt.title("Shannon entropy distribution")
    plt.show()

    x = 2 * (1 + np.arange(10))
    ck = 2**x
    bk = np.sqrt(2 / (m.pi * x)) * 2**x
    # bk = [m.comb(n, int(n / 2)) for n in x]
    rk = [oeis.A000031(n) for n in x]
    # ek = [oeis.A000014(n) for n in x]
    dk = np.zeros(10)
    for i in range(len(x)):
        dk[i] = m.factorial(int(x[i])) / (
            m.factorial(int(x[i] / 2)) * (m.factorial(int(x[i] / 2 + 1)))
        )

    plt.plot(x, ck, label=r"$C_k^{(N)}$")
    plt.plot(x, bk, label=r"$B_k^{(N)}$")
    plt.plot(x, rk, label=r"$R_k^{(N)}$")
    plt.plot(x, dk, label=r"$D_k^{(N)}$")
    # plt.plot(ek)
    plt.title("Number of strings by type and size")
    plt.xlabel("Size")
    plt.ylabel("Number of possible strings")
    plt.legend()
    plt.yscale("log")
    plt.show()
    return


if __name__ == "__main__":
    main(50, 50)
