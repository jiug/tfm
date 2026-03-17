import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def init_pool(n_0: int, n_1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool = np.array(["0", "1"])
    copies = np.array([n_0, n_1])
    indexes = np.array([0, 0])
    return pool, copies, indexes


def combine(pool: np.ndarray, copies: np.ndarray, indexes: np.ndarray) -> None:
    n = np.sum(copies)
    items = np.random.choice(pool, size=2, p=n / copies)

    new_element = items[0] + items[1]
    pool = np.append(pool, new_element)
