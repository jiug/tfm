import argparse
import math as m

import lempel_ziv_complexity as lz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from oeis_sequences import OEISsequences as oeis
from tqdm import tqdm
import secrets

seed = 149240701493537193154751716433558923303
# Uncomment for randomly generated seed
# seed = secrets.randbits(128)
rng = np.random.default_rng(seed)


class AssemblyPool:
    """
    A pool of binary strings that evolves through stochastic combination.

    This class encapsulates a set of binary strings and provides methods for
    evolving them through random concatenation, analyzing their properties,
    and tracking metrics over time.

    Attributes:
        pool (pd.DataFrame): DataFrame containing all elements in the pool with
            their properties and metrics.
        _pending_elements (list): Temporary storage for new elements before batch update.
        _batch_size (int): Number of elements to accumulate before updating the pool.
    """

    def __init__(self, n1: int, n0: int, batch_size: int = 1000) -> None:
        """
        Initialize an AssemblyPool with n1 copies of "1" and n0 copies of "0".

        Args:
            n1 (int): Initial copy number for element "1".
            n0 (int): Initial copy number for element "0".
            batch_size (int, optional): Number of new elements to accumulate before
                updating the pool DataFrame. Defaults to 1000.

        Returns:
            None
        """
        self._pending_elements = []
        self._batch_size = batch_size
        self.pool = self._initialize(n1, n0)

    def _initialize(self, n1: int, n0: int) -> pd.DataFrame:
        """
        Initialize the pool with two seed elements: "1" and "0".

        Args:
            n1 (int): Initial copy number for element "1".
            n0 (int): Initial copy number for element "0".

        Returns:
            pd.DataFrame: Initial pool DataFrame with columns:
                - element: The binary string ("1" or "0")
                - copy_number: Count of this element in the pool
                - size: Length of the string
                - balanced: Whether the string has equal 0s and 1s
                - dyck_word: Whether the string is a valid Dyck word
                - entropy: Shannon entropy of the string
                - assembly: Assembly index from OEIS A003313
                - assembly_efficiency: Ratio of steps to assembly
                - history: Construction history as a string
                - steps: Number of steps taken to create this element
                - lz_comp: Lempel-Ziv complexity
                - has_inversion: Whether the inverted string exists in pool
                - hamming_weight: Number of 1s in the string
                - m-scale_complexity: Multi-scale complexity measured by recursive
                  coarse-graining with window=3
        """
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
            "hamming_weight": [1, 0],
            "m-scale_complexity": [
                multi_scale_complexity("1"),
                multi_scale_complexity("0"),
            ],
        }
        pool = pd.DataFrame(data=initial_pool)
        return pool

    def _update_elements(self) -> None:
        """
        Add pending elements to the main pool DataFrame.

        This method converts the accumulated pending elements into a DataFrame
        and concatenates them with the existing pool, then clears the pending
        elements list.

        Returns:
            None (modifies self.pool in place)
        """
        # Convert the dictionary to DataFrame
        data = pd.DataFrame(self._pending_elements)
        # Add a new observation to the dataframe
        self.pool = pd.concat([self.pool, data], ignore_index=True)
        # Restore pending elements to initial state
        self._pending_elements = []

    def _combine(self, a003313: pd.DataFrame, rng=rng) -> None:
        """
        Combine two random elements from the assembly pool to create a new element.

        This method selects two elements from the pool with probability proportional
        to their copy numbers, concatenates them (with the larger element first),
        and either increments the copy number of an existing matching element or
        adds a new element to the pool.

        Args:
            a003313 (pd.DataFrame): DataFrame containing OEIS sequence A003313 values
                for assembly index calculations.
            rng (np.random.Generator, optional): Random number generator. Defaults to
                the module-level rng.

        Returns:
            None (modifies self.pool and self._pending_elements in place)

        Note:
            The history string tracks the construction tree of each element.
            If the new element already exists in the pool, its copy number is incremented.
            Otherwise, a new element is added to _pending_elements for batch update.
        """
        n = np.sum(self.pool.copy_number)
        # Select only non extinct elements
        non_extinct = self.pool.copy_number > 0
        prob = self.pool.copy_number[non_extinct] / n
        index = rng.choice(
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
            assembly = a003313.iloc[length - 1, 1]
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
                "hamming_weight": hamming_weight(new_element),
                "m-scale_complexity": multi_scale_complexity(new_element),
            }
            self._pending_elements.append(new_observation)

            if len(self._pending_elements) == self._batch_size:
                self._update_elements()
        self._update_elements()
        return

    def evolve(self, steps: int) -> pd.DataFrame:
        """
        Evolve the pool for a given number of steps.

        This method runs the combination process for the specified number of steps,
        periodically recording evolution metrics. The evolution is driven by the
        _combine method which selects and combines elements from the pool.

        Args:
            steps (int): Number of evolution steps to perform.

        Returns:
            pd.DataFrame: DataFrame containing evolution metrics recorded at
                logarithmic intervals. Columns include:
                - count: Total number of non-extinct elements
                - count_balanced: Number of balanced strings (equal 0s and 1s)
                - count_dyck_words: Number of Dyck words
                - max_size: Maximum string length
                - max_steps: Maximum steps to create any element
                - max_lz_comp: Maximum Lempel-Ziv complexity
                - assembly_ceiling: Maximum assembly index
                - ensemble_entropy: Diversity of the pool
                - max_hamming: Maximum Hamming weight (number of 1s)
                - max_multi_scale_complexity: Maxumum multi-scale complexity value in the pool

        Note:
            Metrics are recorded when log2(step) is an integer, resulting in
            measurements at exponentially increasing intervals.
        """
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
                "max_hamming": 1,
                "max_m_scale_comp": 0,
            }
        ]
        a003313 = pd.read_csv("A003313.csv", sep=" ", header=None, engine="python")
        for step in tqdm(range(steps), desc="Calculating iterations"):
            self._combine(a003313)
            non_extinct = self.pool.copy_number > 0
            if np.log2(max(1, step)) % 1 == 0:
                evol_metrics = {
                    "count": len(self.pool[non_extinct]),
                    "count_balanced": np.sum(self.pool[non_extinct].balanced),
                    "count_dyck_words": np.sum(self.pool[non_extinct].dyck_word),
                    "max_size": np.max(self.pool[non_extinct].size),
                    "max_steps": np.max(self.pool[non_extinct].steps),
                    "max_lz_comp": np.max(self.pool[non_extinct].lz_comp),
                    "assembly_ceiling": max(self.pool[non_extinct].assembly),
                    "ensemble_entropy": self.ensemble_entropy(),
                    "max_hamming": np.max(self.pool[non_extinct].hamming_weight),
                    "max_m_scale_comp": np.max(
                        self.pool[non_extinct]["m-scale_complexity"]
                    ),
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

    def modularity(self, idx: int) -> np.ndarray:
        """
        Calculate modularity vector for a given element.

        This method counts how many times each element in the pool appears as a
        substring within the element at the specified index.

        Args:
            idx (int): Index of the element in the pool to analyze.

        Returns:
            np.ndarray: Array where each entry i contains the count of how many
                times pool.element[i] appears as a substring in pool.element[idx].
        """
        string = self.pool.element[idx]
        elements = self.pool.element
        modularity = np.zeros(len(self.pool.element))
        for i in range(len(elements)):
            modularity[i] = string.count(elements[i])
        return modularity

    def ensemble(self) -> None:
        """
        Calculate ensemble probabilities for all elements in the pool.

        This method computes the probability distribution of elements based on
        their copy numbers, storing the result in a new 'ensemble' column.

        Returns:
            None (modifies self.pool in place by adding 'ensemble' column)
        """
        # Select only non-extinct elements
        non_extinct = self.pool.copy_number > 0
        n = np.sum(self.pool.copy_number[non_extinct])
        self.pool["ensemble"] = self.pool.copy_number / n
        return

    def ensemble_entropy(self) -> float:
        """
        Calculate the Shannon entropy of the ensemble distribution.

        This method measures the diversity of the pool by computing the entropy
        of the copy number distribution.

        Returns:
            float: Shannon entropy of the ensemble, in bits.
        """
        self.ensemble()
        return -np.sum(
            self.pool.ensemble[self.pool.copy_number > 0]
            * np.log2(self.pool.ensemble[self.pool.copy_number > 0])
        )


# Functions for string manipulation


def check_word(word: str) -> bool:
    """
    Check if a binary string is a valid Dyck word.

    A Dyck word is a balanced string of parentheses where every prefix has at least
    as many opening parentheses as closing ones. In this implementation, '1' represents
    an opening parenthesis and '0' represents a closing parenthesis.

    Args:
        word (str): Binary string to check (e.g., "1100" or "1010").

    Returns:
        bool: True if the word is a valid Dyck word, False otherwise.

    Note:
        This implementation checks the Dyck word condition by verifying that
        at every position in the string, the count of '1's is >= the count of '0's.
    """
    first = word[0]
    other = str(1 - int(word[0]))
    for _ in range(len(word)):
        if word[:-1].count(first) <= word[:-1].count(other):
            continue
        else:
            return False
    return True


def string_entropy(string: str) -> float:
    """
    Calculate the Shannon entropy of a binary string.

    Args:
        string (str): Binary string to analyze (e.g., "1010").

    Returns:
        float: Shannon entropy in bits. Returns 0 if the string contains only
            one type of character (all 0s or all 1s).

    Note:
        The entropy is calculated as: -p0*log2(p0) - p1*log2(p1)
        where p0 and p1 are the proportions of 0s and 1s respectively.
    """
    n = len(string)
    n0 = string.count("0")
    n1 = n - n0
    p0 = n0 / n
    p1 = n1 / n
    if n0 == 0 or n1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def invert_string(string: str) -> str:
    """
    Invert a binary string by flipping all bits.

    Args:
        string (str): Binary string to invert (e.g., "1010").

    Returns:
        str: Inverted string where all '0's become '1's and vice versa (e.g., "0101").
    """
    return "".join([str(1 - int(x)) for x in string])


def hamming_weight(string: str) -> int:
    """
    Calculate the Hamming weight (number of 1s) of a binary string.

    Args:
        string (str): Binary string to analyze (e.g., "1010").

    Returns:
        int: Count of '1' characters in the string.
    """
    return sum([int(x) for x in list(string)])


def coarse_grain(string: str, window: int = 3) -> tuple[str, str]:
    """
    Coarse-grain a binary string by averaging chunks of size `window`.

    This function divides the string into chunks of the specified window size,
    calculates the average value of each chunk (rounded to nearest integer),
    and returns both the coarse-grained string and a resized version.

    Args:
        string (str): Input binary string (e.g., "1010").
        window (int): Size of the chunks for coarse-graining.

    Returns:
        tuple: (coarse_grained_string, resized_coarse_grained)
            - coarse_grained_string: String of averaged chunks (e.g., "10").
            - resized_coarse_grained: Coarse-grained string resized to original length
              by repeating each bit window times.

    Raises:
        ValueError: If `window` is non-positive.

    Note:
        If the string length is not divisible by window, the string is padded
        with '0's to make it divisible. This may affect the result for strings
        where the remainder is significant.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if len(string) == 0:
        return "", ""

    # Pad the string if necessary (without modifying the original)
    padded_string = string
    remainder = len(string) % window
    if remainder != 0:
        padded_string += (window - remainder) * "0"

    n_chunks = len(padded_string) // window
    coarse_grained_string = ""
    resized_coarse_grained = ""

    for i in range(n_chunks):
        chunk = padded_string[i * window : (i + 1) * window]
        chunk_mean = np.mean([int(x) for x in chunk])
        rounded_mean = np.round(chunk_mean)
        coarse_grained_string += str(int(rounded_mean))
        resized_coarse_grained += str(int(rounded_mean)) * window

    return coarse_grained_string, resized_coarse_grained


def norm_hamming_distance(string1: str, string2: str) -> float:
    """
    Calculate the normalized Hamming distance between two binary strings.

    The Hamming distance is the number of positions at which the corresponding
    characters differ. The normalized distance divides this by the string length.

    Args:
        string1 (str): First binary string.
        string2 (str): Second binary string.

    Returns:
        float: Normalized Hamming distance between 0 and 1.

    Raises:
        ValueError: If the strings have different lengths.
    """
    len1 = len(string1)
    if len1 != len(string2):
        raise ValueError("Strings must be of equal length.")
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter / len1


def multi_scale_complexity(string: str, window: int = 3) -> float:
    """
    Calculate multi-scale complexity using a constant window at all levels.

    This function recursively coarse-grains a binary string and accumulates the
    normalized Hamming distance between each original string and its resized
    coarse-grained version.

    Args:
        string (str): Input binary string to analyze.
        window (int): Constant window size for all coarse-graining levels (default=3).
            Must be positive.

    Returns:
        float: Accumulated normalized Hamming distance across all scales.

    Note:
        - Uses window=3 by default, which avoids rounding ambiguity for binary strings.
        - At each level, the string is coarse-grained using the specified window.
        - The Hamming distance is measured between the original string and the
          resized coarse-grained version (same length as original).
        - The process continues with the non-resized coarse-grained string until
          it becomes shorter than the window size.
        - Distances are accumulated (summed) across all levels.
        - For window=3, possible chunk averages are 0, 1/3, 2/3, 1 which round to 0, 0, 1, 1.

    Raises:
        ValueError: If window is non-positive.
    """
    if window <= 0:
        raise ValueError("Window size must be positive.")

    total_distance = 0.0
    current_string = string

    while len(current_string) >= window:
        # Coarse-grain with constant window
        coarse, resized = coarse_grain(current_string, window)

        # Measure Hamming distance between original and resized
        min_len = min(len(current_string), len(resized))
        if min_len > 0:
            distance = norm_hamming_distance(
                current_string[:min_len], resized[:min_len]
            )
            # Normalize by the fraction of the string that was compared
            total_distance += distance * (min_len / len(current_string))

        # Next level: use the non-resized coarse string
        current_string = coarse

    return total_distance


# Functions for visualization


def evol_graph(evolution: pd.DataFrame) -> None:
    """
    Plot evolution metrics over time.

    This function creates a line plot of various evolution metrics from the
    simulation, showing how the pool properties change over time.

    Args:
        evolution (pd.DataFrame): DataFrame containing evolution metrics with
            time (step) as the index.

    Returns:
        None (displays a matplotlib plot)

    Note:
        All metrics are plotted on a logarithmic y-scale to better visualize
        the exponential growth patterns typical in these simulations.
    """
    x = evolution.index
    for col in [
        "ensemble_entropy",
        "assembly_ceiling",
        "max_steps",
        "max_size",
        "count_dyck_words",
        "count",
        "max_hamming",
        "max_m_scale_comp",
    ]:
        sns.lineplot(data=evolution, x=x, y=col, label=str(col))
    plt.yscale("log")
    plt.xlabel("Simulation step")
    plt.title("Evolution metrics")
    plt.grid(which="both")
    plt.tight_layout()
    plt.legend()
    plt.show()


def total_counts() -> None:
    """
    Plot theoretical counts of different string types by size.

    This function compares the number of possible strings of different types
    (all strings, balanced strings, Dyck words, etc.) as a function of size.

    Returns:
        None (displays a matplotlib plot)

    Note:
        The plot shows:
        - C_k: All possible binary strings of length k (2^k)
        - B_k: Approximation of balanced strings (sqrt(2/pi*k) * 2^k)
        - R_k: Number of Dyck words of length 2k (Catalan numbers)
        - D_k: Exact count of balanced strings
    """
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
    """
    Main function to run the assembly pool simulation.

    This function initializes an AssemblyPool, runs the evolution for the
    specified number of steps, and displays various visualizations of the
    results.

    Args:
        n0 (int): Initial copy number for element "0".
        n1 (int): Initial copy number for element "1".
        steps (int): Number of evolution steps to perform.

    Returns:
        None (displays plots and returns nothing)

    Note:
        The function displays:
        - Evolution metrics over time
        - Pairplot of pool metrics
        - Pairplot of evolution metrics
    """
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
        vars=[
            "copy_number",
            "size",
            "entropy",
            "assembly",
            "lz_comp",
            "steps",
            "hamming_weight",
            "m-scale_complexity",
        ],
        hue="has_inversion",
        diag_kind="kde",
        plot_kws=dict(marker=".", size=2),
    )
    plt.title("Assembly Pool Metrics")
    plt.show()

    sns.pairplot(
        evolution,
        plot_kws=dict(marker=".", size=2),
    )
    plt.title("Evolution Metrics")
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
