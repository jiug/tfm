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
    def __init__(self, n1: int, n0: int, batch_size: int = 1000) -> None:
        self._pending_elements = []
        self._batch_size = batch_size
        self.pool = self._initialize(n1, n0)

    def _initialize(self, n1, n0) -> pd.DataFrame:
        initial_pool = {
            "element": ["1", "0"],
            "copy_number": [n1, n0],
            "size": [1, 1],
            "balanced": [False, False],
            "dyck_word": [False, False],
            "entropy": [0, 0],
            "assembly": [0, 0],
            "assembly_efficiency": [1, 1],
            "history": ["(1)", "(0)"],
            "steps": [0, 0],
            "lz_comp": [1, 1],
            "has_inversion": [True, True],
        }
        pool = pd.DataFrame(data=initial_pool)
        return pool

    def _update_elements(self) -> None:
        # Convert the dictionary to DataFrame
        data = pd.DataFrame(self._pending_elements)
        # Add a new observation to the dataframe
        self.pool = pd.concat([self.pool, data], ignore_index=True)
        # Restore pending elements to initial state
        self._pending_elements = []

    # Function that concatenates two random elements from the assembly pool
    def _combine(self, a003313: pd.DataFrame) -> None:
        n = np.sum(self.pool.copy_number)
        # Select only non extinct elements
        non_extinct = self.pool.copy_number > 0
        prob = self.pool.copy_number[non_extinct] / n
        index = np.random.choice(
            np.arange(len(self.pool.element[non_extinct])), size=2, p=prob
        )
        element0, element1 = self.pool.element[index]

        # Logic to keep the biggest element always first
        if len(element0) >= len(element1):
            new_element = element0 + element1
            history_big = self.pool.history[index[0]]
            history_small = self.pool.history[index[1]]
        else:
            new_element = element1 + element0
            history_big = self.pool.history[index[1]]
            history_small = self.pool.history[index[0]]

        # Fin the new element in the assembly pool
        idx = self.pool.loc[self.pool.element == new_element].index

        # If the new element is not idx will be an empty array
        if idx.size > 0:

            # Increase the copy number of the existing element
            if history_big.count(history_small) > 0:
                self.pool.at[idx[0], "copy_number"] += 1
            else:
                self.pool.at[idx[0], "copy_number"] = (
                    self.pool.loc[index[0], "steps"]
                    + self.pool.loc[index[1], "steps"]
                    + 1
                )
            return
        else:
            history = f"({history_big + history_small})"

            # Add new element to the assembly pool
            # Same structure as the __init__ dictionary
            length = len(new_element)
            inverted = invert_string(new_element)
            balanced = new_element.count("0") == new_element.count("1")
            assembly = a003313.iloc[length, 1]
            steps = np.max(self.pool.steps[index]) + 1
            dyck_word = check_word(new_element)
            new_observation = {
                "element": new_element,
                "copy_number": 1,
                "size": len(new_element),
                "balanced": balanced,
                "dyck_word": balanced and dyck_word,
                "entropy": string_entropy(new_element),
                "assembly": assembly,
                "assembly_efficiency": steps / assembly,
                "history": history,
                "steps": steps,
                "lz_comp": lz.lempel_ziv_complexity(new_element),
                "has_inversion": (
                    True if sum(self.pool.element.isin([inverted])) > 0 else False
                ),
            }
            self._pending_elements.append(new_observation)

            if len(self._pending_elements) == self._batch_size:
                self._update_elements()
        self._update_elements()
        return

    def evolve(self, steps: int) -> pd.DataFrame:
        init_data = [
            {
                "count": len(self.pool),
                "count_balanced": np.sum(self.pool.balanced),
                "count_dyck_words": 0,
                "max_size": np.max(self.pool.size),
                "max_steps": np.max(self.pool.steps),
                "max_lz_comp": np.max(self.pool.lz_comp),
                "assembly_ceiling": 0,
                "ensemble_entropy": 0,
            }
        ]
        a003313 = pd.read_csv("A003313.csv", sep=" ", header=None, engine="python")
        for step in tqdm(range(steps), desc="Calculating iterations"):
            non_extinct = self.pool[self.pool.copy_number > 0]
            self._combine(a003313)
            if step % 10 == 0:
                evol_metrics = {
                    "count": len(non_extinct),
                    "count_balanced": np.sum(non_extinct.balanced),
                    "count_dyck_words": np.sum(non_extinct.dyck_word),
                    "max_size": np.max(non_extinct.size),
                    "max_steps": np.max(non_extinct.steps),
                    "max_lz_comp": np.max(non_extinct.lz_comp),
                    "assembly_ceiling": max(non_extinct.assembly),
                    "ensemble_entropy": self.ensemble_entropy(),
                }
                init_data.append(evol_metrics)
        evolution = pd.DataFrame(init_data)
        # if step % 500 == 0:
        #     """
        #     Selection Rule I
        #     Remove any string with a hamming_weight (n_1) > sqrt(size)
        #     """
        #     weights = self.pool.element.apply(hamming_weight)
        #     mask = self.pool[weights > np.sqrt(self.pool.size)].index
        #     self.pool.loc[mask, "copy_number"] = 0
        #     print(f"There were {len(mask)} species extinct")
        #
        #     """
        #     Selection Rule II
        #     """
        #
        return evolution

    # Counts how many times smaller strings 'fit' into a given reference string
    def modularity(self, idx: int) -> np.ndarray:
        string = self.pool.element[idx]
        elements = self.pool.element
        modularity = np.zeros(len(self.pool.element))
        for i in range(len(elements)):
            modularity[i] = string.count(elements[i])
        return modularity

    def ensemble(self) -> None:
        # Select only non-extinct elements
        non_extinct = self.pool.copy_number > 0
        n = np.sum(self.pool.copy_number[non_extinct])
        self.pool["ensemble"] = self.pool.copy_number / n
        return

    def ensemble_entropy(self) -> float:
        self.ensemble()
        return -np.sum(
            self.pool.ensemble[self.pool.copy_number > 0]
            * np.log2(self.pool.ensemble[self.pool.copy_number > 0])
        )


# Functions for string manipulation


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


def invert_string(string: str) -> str:
    return "".join([str(1 - int(x)) for x in string])


def hamming_weight(string: str) -> int:
    return sum([int(x) for x in list(string)])


def coarse_grain(string: str, window: int) -> str:
    n_chunks = len(string) // window
    chunks = np.zeros(n_chunks)
    coarse_grained_string = ""
    if len(string) % window != 0:
        n_chunks += 1
        string += (window - len(string) % window) * "0"

    for i in range(n_chunks):
        chunks[i] = np.round(
            np.mean([int(x) for x in string[i * window : (i + 1 * window)]])
        )
        coarse_grained_string += str(int(chunks[i]))
    print(chunks)
    return coarse_grained_string


# Functions for graph generation


def evol_graph(evolution: pd.DataFrame) -> None:
    x = np.arange(len(evolution))

    i = 0
    for col in ["ensemble_entropy", "assembly_ceiling", "max_steps"]:
        sns.lineplot(data=evolution, x=x, y=col, label=str(col))
        i += 1
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Simulation step")
    plt.title("Evolution metrics")
    plt.grid(which="major")
    plt.tight_layout()
    plt.show()


def total_counts() -> None:
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


def main(n0: int, n1: int, steps: int) -> None:
    ap = AssemblyPool(n0, n1)
    evolution = ap.evolve(steps)
    evol_graph(evolution)

    # Graphs
    # sns.scatterplot(data=ap.pool, x="size", y="copy_number", hue="entropy")
    # plt.grid(which="both", linestyle=":")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlabel("Size")
    # plt.ylabel("Copy number")
    # plt.title("Copies of bitstrings by length")
    # plt.legend()
    # plt.show()
    #
    # ap.pool["log_copy"] = np.log1p(ap.pool["copy_number"])
    # sns.scatterplot(data=ap.pool, x="size", y="entropy", hue="steps")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.grid(which="both")
    # plt.show()
    #
    # sns.scatterplot(data=ap.pool, x="lz_comp", y="entropy", hue="balanced")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylabel("Entropy")
    # plt.xlabel("Lempel-Ziv Complexity")
    # plt.title("Complexity vs Entropy")
    # plt.show()
    #
    # sns.scatterplot(data=ap.pool, x="size", y="lz_comp")
    # plt.ylabel("Complexity")
    # plt.xlabel("Size")
    # plt.title("Correlation between size and comp")
    # plt.show()
    #
    sns.pairplot(
        ap.pool,
        vars=["copy_number", "size", "entropy", "assembly", "lz_comp", "steps"],
        hue="dyck_word",
        diag_kind="kde",
        plot_kws=dict(marker=".", size=2),
    )
    plt.title("Assembly Pool Metrics")
    plt.show()

    sns.pairplot(
        evolution,
        plot_kws=dict(marker=".", size=2),
    )
    plt.title("Assembly Pool Metrics")
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
    args = parser.parse_args()

    main(args.n1, args.n0, args.steps)
