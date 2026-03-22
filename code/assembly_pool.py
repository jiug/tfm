import argparse
import math as m

import lempel_ziv_complexity as lz
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
            "size": np.array([1, 1]),
            "balanced": np.array([False, False]),
            "dyck_word": np.array([False, False]),
            "entropy": np.array([0, 0]),
            "assembly": np.array([0, 0]),
            "history": np.array(["1", "0"]),
            "steps": np.array([0, 0]),
            "lz_comp": np.array([1, 1]),
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
            balanced = new_element.count("0") == new_element.count("1")
            dyck_word = np.array([check_word(new_element)])
            new_observation = {
                "element": np.array([new_element]),
                "copy_number": np.array([1]),
                "size": np.array(len(new_element)),
                "balanced": balanced,
                "dyck_word": balanced and dyck_word,
                "entropy": np.array([string_entropy(new_element)]),
                "assembly": np.array([0]),
                "history": np.array(history),
                "steps": np.array(np.max(self.pool.steps[index]) + 1),
                "lz_comp": lz.lempel_ziv_complexity(new_element),
            }
            # Convert the dictionary to DataFrame
            data = pd.DataFrame(data=new_observation)
            # Add a new observation to the dataframe
            self.pool = pd.concat([self.pool, data], ignore_index=True)
        return

    def evolve(self, steps: int) -> pd.DataFrame:
        init_data = {
            "count": [len(self.pool)],
            "count_balanced": [np.sum(self.pool.balanced)],
            "count_dyck_words": [np.sum(self.pool.dyck_word)],
            "max_size": [np.max(self.pool.size)],
            "max_lz_comp": [np.max(self.pool.lz_comp)],
        }
        evolution = pd.DataFrame(init_data)

        for i in tqdm(range(steps), desc="Calculating iterations"):
            self.combine()
            evol_measures = {
                "count": [len(self.pool)],
                "count_balanced": [np.sum(self.pool.balanced)],
                "count_dyck_words": [np.sum(self.pool.dyck_word)],
                "max_size": [np.max(self.pool.size)],
                "max_lz_comp": [np.max(self.pool.lz_comp)],
            }
            evolution = pd.concat(
                [evolution, pd.DataFrame(evol_measures)], ignore_index=True
            )
        return evolution


def check_word(word: str) -> bool:
    first = word[0]
    other = str(1 - int(word[0]))
    for _ in range(len(word)):
        if word[:-1].count(first) <= word[:-1].count(other):
            continue
        else:
            return False
    return True


def string_entropy(string: str) -> float:
    n = len(string)
    n0 = string.count("0")
    n1 = n - n0
    p0 = n0 / n
    p1 = n1 / n
    if n0 == 0 or n1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def evol_graph(evolution):
    x = np.arange(len(evolution))
    metrics = [
        "Element number",
        "Balanced number",
        "Dyck Word number",
        "Maximum size",
        "Maximum L-Z Complexity",
    ]
    i = 0
    for col in evolution.columns:
        sns.lineplot(data=evolution, x=x, y=col, label=metrics[i])
        i += 1
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Simulation step")
    plt.title("Evolution metrics")
    plt.grid(which="both")
    plt.tight_layout()
    plt.show()


def main(n0: int, n1: int, steps: int) -> None:
    ap = AssemblyPool(n0, n1)
    evolution = ap.evolve(steps)
    evol_graph(evolution)
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
    sns.scatterplot(data=ap.pool, x="size", y="entropy", hue="steps")
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
    parser = argparse.ArgumentParser(
        prog="assembly_pool",
        usage="%(prog)s initial-1s initial-0s evolution-steps ",
        description="Script that evolves pool of binary strings following simple stochastic combination rules",
        epilog="Example:  ipython assembly_pool.py 10 10 5000",
    )
    parser.add_argument("n1", type=int, help="1's initial copy number")
    parser.add_argument("n0", type=int, help="0's initial copy number")
    parser.add_argument("steps", type=int, help="Number of evolution steps")
    # parser.add_argument("-s", type=bool, help="Use a hardcoded seed")
    args = parser.parse_args()

    main(args.n1, args.n0, args.steps)
