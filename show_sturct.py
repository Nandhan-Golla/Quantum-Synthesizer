
path = '/media/nandhan/Alpha-Card/Medical-Database/UP000000799_192222_CAMJE_v4/AF-O52908-F1-model_v4.pdb.gz'
import gzip
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load structure
parser = PDBParser(QUIET=True)
with gzip.open(path, 'rt') as f:
    structure = parser.get_structure("AF_model", f)

# Collect atom coordinates
x, y, z = [], [], []
for atom in structure.get_atoms():
    coord = atom.get_coord()
    x.append(coord[0])
    y.append(coord[1])
    z.append(coord[2])

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=10, alpha=0.6)
ax.set_title("Protein Structure (Atoms)")
plt.show()

