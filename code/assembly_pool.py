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
            "history": ["1", "0"],
            "steps": [0, 0],
            "lz_comp": [1, 1],
            "has_inversion": [True, True],
        }
        pool = pd.DataFrame(data=initial_pool)
        return pool

    def _update_elements(self):
        # Convert the dictionary to DataFrame
        data = pd.DataFrame(self._pending_elements)
        # Add a new observation to the dataframe
        self.pool = pd.concat([self.pool, data], ignore_index=True)
        # Restore pending elements to initial state
        self._pending_elements = []

    # Function that concatenates two random elements from the assembly pool
    def combine(self, a003313: pd.DataFrame) -> None:
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
            length = len(new_element)
            inverted = invert_string(new_element)
            balanced = new_element.count("0") == new_element.count("1")
            dyck_word = check_word(new_element)
            new_observation = {
                "element": new_element,
                "copy_number": 1,
                "size": len(new_element),
                "balanced": balanced,
                "dyck_word": balanced and dyck_word,
                "entropy": string_entropy(new_element),
                "assembly": a003313.iloc[length, 1],
                "history": history,
                "steps": np.max(self.pool.steps[index]) + 1,
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
        init_data = {
            "count": [len(self.pool)],
            "count_balanced": [np.sum(self.pool.balanced)],
            "max_size": [np.max(self.pool.size)],
            "max_lz_comp": [np.max(self.pool.lz_comp)],
            "assembly_ceiling": [0],
            "ensenble_entropy": [self.ensemble_entropy()],
            "innovation": [0],
            "extinction": [0],
        }
        evolution = pd.DataFrame(init_data)
        a003313 = pd.read_csv("A003313.csv", sep=" ", header=None, engine="python")
        for _ in tqdm(range(steps), desc="Calculating iterations"):
            innovation_event = max(evolution.innovation)
            extinction_event = max(evolution.extinction)
            count = len(self.pool.element)
            self.combine(a003313)
            if len(self.pool.element) > count:
                innovation_event += 1
            elif len(self.pool.element) < count:
                extinction_event += 1
            evol_measures = {
                "count": [len(self.pool)],
                "count_balanced": [np.sum(self.pool.balanced)],
                "count_dyck_words": [np.sum(self.pool.dyck_word)],
                "max_size": [np.max(self.pool.size)],
                "max_lz_comp": [np.max(self.pool.lz_comp)],
                "assembly_ceiling": [max(self.pool.assembly)],
                "ensemble_entropy": [self.ensemble_entropy()],
                "innovation": [innovation_event],
                "extinction": [extinction_event],
            }
            evolution = pd.concat(
                [evolution, pd.DataFrame(evol_measures)], ignore_index=True
            )
        return evolution

    def modularity(self, idx: int) -> np.ndarray:
        string = self.pool.element[idx]
        elements = self.pool.element
        modularity = np.zeros(len(self.pool.element))
        for i in range(len(elements)):
            modularity[i] = string.count(elements[i])
        return modularity

    def ensemble(self) -> None:
        n = np.sum(self.pool.copy_number)
        self.pool["ensemble"] = self.pool.copy_number / n
        return

    def ensemble_entropy(self) -> float:
        self.ensemble()
        return -np.sum(self.pool.ensemble * np.log2(self.pool.ensemble))


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


def coarse_grain(string: str, window: int) -> str:
    chunks = [
        string[x * window : min((x + 1) * window, len(string))]
        for x in range(len(string) // window + 1)
    ]

    averages = [
        np.round(np.mean(np.array([int(i) for i in list(chunk)]))) for chunk in chunks
    ]
    print(averages)
    coarse_grained_string = ""
    for i in range(len(averages)):
        coarse_grained_string += f"{int(averages[i])}"
    coarse_grained_string = coarse_grained_string[: len(string)]
    return coarse_grained_string


def evol_graph(evolution: pd.DataFrame) -> None:
    x = np.arange(len(evolution))
    metrics = [
        "Element number",
        "Balanced number",
        "Dyck Word number",
        "Maximum size",
        "Maximum L-Z Complexity",
        "Assembly Ceiling",
        "Ensemble Entropy",
        "Hello",
    ]
    i = 0
    for col in evolution.columns:
        sns.lineplot(data=evolution, x=x, y=col, label=evolution.columns[i])
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

    sns.scatterplot(data=ap.pool, x="lz_comp", y="entropy")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Entropy")
    plt.xlabel("Lempel-Ziv Complexity")
    plt.title("Complexity vs Entropy")
    plt.show()

    sns.scatterplot(data=ap.pool, x="size", y="lz_comp")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.ylabel("Complexity")
    plt.xlabel("Size")
    plt.title("Correlation between size and comp")
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
