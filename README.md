# tfm
code and data for my computational physics master's thesis

## `constant_components.py`
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

### Default output
The script prints the size of the biggest element in the set and an upper bound for it's assembly index calculated incrementally (not accounting for shortcuts in the assembly history).

<img src="https://github.com/jiug/tfm/blob/master/figs/gset_n100_d6_t10_b15.png" alt="Output of the script" width="500" height="500">

**Figure 1:** Result of running the example code provided above. 

## `binary_trees.py`

### Required arguments
- `n: int` total number of trees in the forest
- `probability: int` probability that a leaf will split when evaluated
- `max_depth` maximum number of levels the algorithm will output
- `new_leaves: int` number of new leaves created on each split 
- `bonds_per_iteration: int` New bonds created on each iteration (to be merged with iterations in later commits)
  
### [options]
- `-g --graph: bool` If true represents the set of graphs (better for smaller `set_size` values)
- `-s --seed: bool` If true uses a hardcoded seed

### Default output

<img src="https://github.com/jiug/tfm/blob/master/figs/dyck_paths.png" alt="Output of the script" width="500" height=auto>
