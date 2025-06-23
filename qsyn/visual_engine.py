from Bio.PDB import PDBParser
import matplotlib.pyplot as plt 
import gzip
import logging

def show_sturcture(path):
    praser = PDBParser(QUIET=True)
    with gzip.open(path, 'rt') as handle:
        strct = praser.get_structure("AF_model", handle)

    x, y, z = [], [], []
    for atom in strct.get_atoms():
        cord = atom.get_coord()
        x.append(cord[0])
        y.append(cord[1])
        z.append(cord[2])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=10, alpha=0.6)
    ax.set_title('Protein Structure (Atomic Scaling)')
    plt.show()