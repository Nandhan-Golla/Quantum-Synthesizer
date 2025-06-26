from Bio.PDB import PDBParser

def pdb_to_seq(path):
    praser = PDBParser
    data = praser.get_structure(path)
    ## ---- ##