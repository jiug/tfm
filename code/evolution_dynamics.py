import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import assembly_pool as ap


def main():

    n_total = np.array([10, 50, 100, 500, 1000])
    steps_total = 4 * n_total
    evol_total = [[0, 0, 0, 0, 0] for _ in range(5)]
    print(evol_total)
    for i in range(len(n_total)):
        n = n_total[i]
        for j in range(len(steps_total)):
            steps = steps_total[j]
            run = ap.AssemblyPool(n / 2, n / 2, 500)
            print(i, j)
            evol_total[i][j] = {f"{i}-{j}": run.evolve(steps)}
    return evol_total


if __name__ == "__main__":
    main()
