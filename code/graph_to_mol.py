import numpy as np
import constant_components as cc
import igraph as ig
from rdkit import Chem
from rdkit.Chem import AllChem
import assembly_theory as at

gset, gsizes, gassembly = cc.main(100,4,200,np.random.default_rng(123456789))

def graph_to_mol(graph: ig.Graph, atom_type = '*') :
    """Convert igraph graph to RDKit mol object"""
    # Create empty editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for i in range(graph.vcount()):
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)

    # Add bonds
    for edge in graph.es:
        mol.AddBond(edge.source, edge.target, Chem.BondType.SINGLE)

    # Convert to regular molecule
    mol = mol.GetMol()
    return mol

