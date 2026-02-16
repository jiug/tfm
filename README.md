# tfm
code and data for my computational physics master's thesis
## Code execution
`python3 constant_components set_size max_degree iterations bonds_per_iteration [options]`

### Required arguments
- `set_size: int` Total number of graphs in the set
- `max_degree: int` Controls the maximum number of edges to/from any node in the set
- `iterations: int` Iterations of the recombination method
- `bonds_per_iteration: int` New bonds created on each iteration (to be merged with iterations in later commits)


### [options]
- `-g --graph: bool` If true represents the set of graphs (better for smaller `set_size` values)
- `-s --seed: bool` If true uses a hardcoded seed


To execute the code run `python3 constant_components.py 100 6 10 15 -g True -s True`
More information could be found running `python3 constant_components.py --help`

## Default output
The script prints the size of the biggest element in the set and an upper bound for it's assembly index calculated incrementally (not accounting for shortcuts in the assembly history).

![Output of the script](https://github.com/jiug/tfm/blob/master/figs/gset_n100_d6_t10_b15.png "Output for N=100, max_degree=6, t=10, b=15")
**Figure1:** Result of running the example code provided above. 
