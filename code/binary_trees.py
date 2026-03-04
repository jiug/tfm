import argparse
import numpy as np
from tqdm import tqdm
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
            if rand <= p:
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

def digitize(word:str)->tuple[str,np.ndarray]:
    #Replace '(' for 1 and ')' for -1
    replaced = word.replace('(', '2').replace(')','0').replace('X','')
    integers = np.array([int(x)-1 for x in list(replaced)])
    return replaced, integers 

def mountains(digits:np.ndarray)->np.ndarray:
    profile = np.zeros(len(digits)+1)
    for i in range(len(digits)):
        profile[i+1] = sum(digits[:i+1])
    return profile

# Coarse graining function that obtains each i-th entry from the average of
# the [w*i,...,w*(i+1)] entries of the digitized Dyck array. 
def coarse_grain(word,window:int)->np.ndarray:
    if type(word) == str:
        _, digits = digitize(word)
    else:
        digits = word
    length = len(digits)
    blocks = length//window
    last_block = length%window
    coarse_grained = np.zeros(blocks + 1) if last_block > 0 else np.zeros(blocks) 
    for i in range(blocks-1):
        coarse_grained[i] = np.mean(digits[i*window:(i+1)*window])
    return  coarse_grained 


def main(n:int,prob:float, max_depth:int, new_leaves: int, g:bool = False)-> None:    
        
    roots, lengths, depths = init_forest(n)
    forest = []
    for i in tqdm(range(len(roots)), desc='Growing forest'):
        tree, depth = grow(roots[i], prob, max_depth , new_leaves)
        lengths[i] = len(tree)
        depths[i] = depth +1
        forest.append(tree)
    if g:
        plt.hist(depths)
        plt.yscale('log')
        plt.grid(which = 'major', linestyle = ':')
        plt.title(rf'Dyck word depths ($n$={n}, $p$={prob}, $d$={max_depth}, $k$={new_leaves})')
        # plt.xlim([0.5, max(lengths)+0.5])
        plt.show()

    max_length = np.argmax(lengths)
    max_depth = np.argmax(depths)
    _, digits_depth = digitize(forest[max_depth])
    _, digits_length= digitize(forest[max_length])
    mountain_length = mountains(digits_length)
    mountain_depth = mountains(digits_depth)

    plt.plot(mountain_depth,label=f'Deepest l={lengths[max_depth]}, d={max(depths)}')
    plt.plot(mountain_length,label=f'Longest l={max(lengths)}, d={depths[max_length]}')
    plt.title(f'Dyck paths')
    plt.legend()
    plt.show()

    print(coarse_grain(digits_length,5))


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
