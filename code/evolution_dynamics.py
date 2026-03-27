import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import assembly_pool as ap


def main():

    n_total = np.array([50, 100, 500, 1000, 1500])
    copies = 10
    steps = 8000
    evol_total = [[0, 0, 0, 0, 0] for _ in range(copies)]
    for c in range(copies):
        for i in range(len(n_total)):
            n = n_total[i]
            print(f"N: {n_total[i]},  {steps} steps")
            run = ap.AssemblyPool(n / 2, n / 2, 500)
            evol_total[c][i] = {f"{i}": run.evolve(steps)}
    return evol_total


if __name__ == "__main__":
    main()
