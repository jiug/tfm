import numpy as np
import re

forest = r'()' 

for i in range(6):
    # Find all the non-terminal leaves '()'
    expression = r'\(\)'

    # If the leaf is to split it adds n new parenthesis (default 2)
    replacement_split = r'()'*2

    # If it doesn't split turns into a terminal leaf (no more splitting) 
    replacement_end = r'X'

    matches = re.finditer(expression, forest)   
    result = '' 
    shift = 0 # Accounts for the added characters on each iteration
    for match in matches:
        start, end = match.span()
        rand = np.random.random()
        if rand >= 0.5:
            result = forest[:start+1+shift] + replacement_split+ forest[start+1+shift:]
            shift += 4
        else: 
            result = forest[:start+1+shift] + replacement_end + forest[start+1+shift:]
            shift += 1
        forest = result
        
# Checks if the string is a well formed Dyck word. 
# At any point in the chain the number of '(' has to be <= the number of ')'
# Removing the last element we have a strict inequality
well_formed = False

for i in range(len(forest)):
    if forest[:-1].count('(')<= forest[:-1].count(')'):
        continue
    else:
        well_formed = True
if not well_formed:
    raise ValueError('Not a Dyck Word')
else:
    print("It's a Dyck Word")

print(forest[:-1].count('('), forest[:-1].count(')'), forest.count('X'))
print(forest)

