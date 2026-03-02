import argparse
import numpy as np
import matplotlib.pyplot as plt
import re


def init_forest(n:int)->tuple[ np.ndarray,np.ndarray, np.ndarray ]:
    forest = np.array(['()']*n)
    lengths = np.zeros(n)
    depths = np.zeros(n)
    return forest, lengths, depths

def grow(tree:str, p:float,d:int, k:int)->tuple[str, int]:
    for i in range(d):
        # Find all the non-terminal leaves '()'
        expression = r'\(\)'

        # If the leaf is to split it adds n new parenthesis (default 2)
        replacement_split = r'()'*k

        # If it doesn't split turns into a terminal leaf (no more splitting) 
        replacement_end = r'X'

        matches = re.finditer(expression, tree)   
        result = '' 
        shift = 0 # Accounts for the added characters on each iteration
        if matches == []:
            break 
        for match in matches:
            start, _ = match.span()
            rand = np.random.random()
            if rand >= p:
                result = tree[:start+1+shift] + replacement_split+ tree[start+1+shift:]
                # The shift depends on the number of new leaves
                shift += 2*k 
                depth = i+1
            else: 
                result = tree[:start+1+shift] + replacement_end + tree[start+1+shift:]
                shift += 1
                depth = i
            tree = result
    return tree, depth 

# Checks if the string is a well formed Dyck word. 
# At any point in the chain the number of '(' has to be <= the number of ')'
# Removing the last element we have a strict inequality
def check_word(word:str)->None:
    well_formed = False
    for _ in range(len(word)):
        if word[:-1].count('(')<= word[:-1].count(')'):
            continue
        else:
            well_formed = True
    if not well_formed:
        raise ValueError('Not a Dyck Word')
    else:
        print("It's a Dyck Word")


def main(n:int,prob:float, max_depth:int, new_leaves: int, g:bool = False)-> None:    
        
    forest, lengths, depths = init_forest(n)
    for i in range(len(forest)):
        tree, depth = grow(forest[i], prob, max_depth , new_leaves)
        lengths[i] = len(tree)
        depths[i] = depth +1
    # print(tree[:-1].count('('), tree[:-1].count(')'), tree.count('X'))
    if g:
        plt.hist(depths)
        plt.yscale('log')
        plt.grid(which = 'major', linestyle = ':')
        plt.title(rf'Dyck word depths ($n$={n}, $p$={prob}, $d$={max_depth}, $k$={new_leaves})')
        # plt.xlim([0.5, max(lengths)+0.5])
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    prog="binary_trees",
    usage="%(prog)s size probability max_depth new_leaves [options]",
    description="Script that generates a forest of binary trees with a probabilistic growth rule",
    epilog="Example:  python3 binary_trees.py 10000 0.5 6 2 -g True",
)
    parser.add_argument("forest_size", type=int, help="Size of the set")
    parser.add_argument(
        "probability", type=float, help="Probability of splitting a leaf"
    )
    parser.add_argument(
        "max_depth", type=int, help="Maximum tree depth"
    )
    parser.add_argument(
        "new_leaves", type=int, help="Number of new leaves created at each split"
    )
    parser.add_argument(
        "-g",
        type=bool,
        help="Toggle for showing a histogram of tree lengths", 
    )
    # parser.add_argument("-s", type=bool, help="Use a hardcoded seed")
    args = parser.parse_args()

    main(args.forest_size, args.probability, args.max_depth, args.new_leaves ,args.g)
