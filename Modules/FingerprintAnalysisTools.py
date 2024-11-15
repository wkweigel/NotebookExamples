from itertools import combinations
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from ipywidgets import interact
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate
import matplotlib.pyplot as plt
import numpy as np

#Define a function to get various fingerprints from a smiles
def get_torsion_fp(mol, nBits=1024):
    # Create the fingerprint generator for topological torsion fingerprints
    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)

    # Get the topological torsion fingerprint
    fp = ttgen.GetFingerprint(mol)

    # Convert the fingerprint object to a NumPy array with dtype int64
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

def get_atompair_fp(mol, nBits=1024):
    # Get the atom pair fingerprint
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)

    # Convert the fingerprint object to a NumPy array with dtype int64
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

def get_avalon_fp(mol, nBits=1024):
    # Get the avalon fingerprint
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)

    # Convert the fingerprint object to a NumPy array with dtype int64
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

def get_rdkit_fp(mol, nBits=1024):
    # Create the fingerprint generator for RDKit fingerprints
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nBits)

    # Get the RDKit fingerprint
    fp = rdkgen.GetFingerprint(mol)

    # Convert the RDKit fingerprint object to a NumPy array with dtype int64
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

def get_fMorgan_fp(mol, nBits=1024):
    # Get the feature Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nBits, useFeatures=True)

    # Convert the RDKit fingerprint object to a NumPy array
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

def get_ecfp_fp(mol, nBits=1024):
    # Get the regular Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nBits)

    # Convert the RDKit fingerprint object to a NumPy array
    np_fp = np.zeros((1,), dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    return np_fp

#Get a 2D pharmacophore fingerprint from the input mol using the Gobbi feature definitions
def get_gobbi_fp(mol, nBits=1024):
    # Generate the Pharm2D fingerprint
    fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    def hash_function(bit_index, bits):
        return hash(bit_index) % bits

    # Create a 1024-bit array initialized to zero
    folded_fp = np.zeros(nBits, dtype=int)

    # Fold the sparse fingerprint down to 1024 bits
    for bit_index in fp.GetOnBits():
        folded_fp[hash_function(bit_index, nBits)] = True
    return folded_fp

def get_single_FP_from_smiles(smiles, fp_type='ecfp', nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if 'rdkit' in fp_type:
        fp = get_rdkit_fp(mol, nBits)
    if 'fMorgan' in fp_type:
        fp = get_fMorgan_fp(mol, nBits)
    if 'gobbi' in fp_type:
        fp = get_gobbi_fp(mol, nBits)
    if 'ecfp' in fp_type:
        fp = get_ecfp_fp(mol, nBits)
    if 'torsion' in fp_type:
        fp = get_torsion_fp(mol, nBits)
    if 'avalon' in fp_type:
        fp = get_avalon_fp(mol, nBits)
    if 'atompair' in fp_type:
        fp = get_atompair_fp(mol, nBits)
    return fp

# Define a function that will pictorially represent two input bitstings for comparison
def display_bitstrings(bitstring1, bitstring2, id1, id2):
    # Convert the bitstrings to numpy arrays of integers
    bits1 = np.array([int(bit) for bit in bitstring1])
    bits2 = np.array([int(bit) for bit in bitstring2])

    # Compute the similarity grid by comparing corresponding bits
    similarity_grid = np.equal(bits1, bits2).astype(int)

    # Display the bitstrings and the similarity grid
    fig, axs = plt.subplots(3, 1, figsize=(12, 3))
    axs[0].imshow(bits1.reshape(1, -1), cmap='Reds', aspect='auto')
    axs[0].set_title(id1)
    axs[0].axis('off')

    axs[1].imshow(bits2.reshape(1, -1), cmap='Blues', aspect='auto')
    axs[1].set_title(id2)
    axs[1].axis('off')

    axs[2].imshow(similarity_grid.reshape(1, -1), cmap='gray', aspect='auto')
    axs[2].set_title('Difference')
    axs[2].axis('off')

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4)
    plt.show()
