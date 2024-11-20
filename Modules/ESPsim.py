import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdDistGeom
from espsim import GetEspSim, GetShapeSim
import py3Dmol
import tqdm
import secrets
from copy import deepcopy

class Conformer:
    '''A class for generating molecule conformers.'''
    def __init__(self, smiles, n_confs, seed = 0xf00d):
        # Define the conformer generator
        p = AllChem.ETKDGv3()
        if seed == 'random':
            p.randomSeed = secrets.token_hex(3)
        else:
            p.randomSeed = seed

        # Create molecule and add hydrogens
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Generate conformers
        conformer_ids = AllChem.EmbedMultipleConfs(mol, n_confs, p)

        # Get MMFF properties for the molecule
        mmff = AllChem.MMFFGetMoleculeProperties(mol)

        self.mol = mol
        self.mmff = mmff
        self.conformer_ids = conformer_ids



def get_real_conformer(mol):
    """
    Generate conformers until a successful 3D conformation is generated.
    """
    conf_failure=0
    comf_attempts=0
    while conf_failure==0:
        rdDistGeom.EmbedMolecule(mol)
        conf_failure=mol.GetNumConformers()
        comf_attempts+=1
        print(str(comf_attempts) + " failed conformer generations. Reattempting...")
        if conf_failure==1:
            break
    return(mol)

def isolate_conformer(mol, conf_id:int):
    """
    Retrives a specific conformer from a mol and returns it as an individual mol conformer.

    Parameters:
        mol (Mol): RDKit Mol object with embedded conformers.
        conf_id (int): The id corresponding to the desired conformer to return.

    Returns:
        new_mol (Mol): RDKit Mol object for the specied conf_id.
    """
    conformer = mol.GetConformer(conf_id)

    # Create a new molecule and add the conformer's coordinates
    new_mol = Chem.Mol(mol)
    new_conformer = Chem.Conformer(mol.GetNumAtoms())
    new_conformer.SetId(0)  # Set the ID of the new conformer
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(atom_idx)
        new_conformer.SetAtomPosition(atom_idx, pos)

    # Replace the new molecule's conformer with the extracted conformer
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(new_conformer)
    return new_mol

def get_confs_constant_seed(query_smi:str, ref_mol):
    # Generate conformers for the query smiles
    query_confs = Conformer(query_smi, 100)
    ref_mmff = AllChem.MMFFGetMoleculeProperties(ref_mol)
    conf_scores = []
    conf_ids = []
    mol_confs = [conf.GetId() for conf in query_confs.mol.GetConformers()]

    # Align and score conformers using the reference mol
    for conf_id in mol_confs:
        O3A = rdMolAlign.GetO3A(query_confs.mol, ref_mol, query_confs.mmff, ref_mmff, conf_id, 0)
        O3A.Align()
        conf_scores.append(O3A.Score())
        conf_ids.append(conf_id)
    return conf_scores, conf_ids, query_confs

def get_confs_random_seed(query_smi:str, ref_mol):
    # Generate conformers for the query smiles
    query_confs = Conformer(query_smi, 50, seed = 'random')
    ref_mmff = AllChem.MMFFGetMoleculeProperties(ref_mol)
    conf_scores = []
    conf_ids = []
    mol_confs = [conf.GetId() for conf in query_confs.mol.GetConformers()]

    # Align and score conformers using the reference mol
    for conf_id in mol_confs:
        O3A = rdMolAlign.GetO3A(query_confs.mol, ref_mol, query_confs.mmff, ref_mmff, conf_id, 0)
        O3A.Align()
        conf_scores.append(O3A.Score())
        conf_ids.append(conf_id)
    return conf_scores, conf_ids, query_confs
    

def get_esp_similarity_score(query_smi:str, ref_mol):
    """
    Calculates espsim similarity score data for an input smiles string with a reference mol.

    Parameters:
        query_smi (string): The smiles string to use for scoring.
        ref_mol (Mol): Reference molecule (RDKit Mol object).

    Returns:
        best_conf (Mol): RDKit Mol object for the closest aligned conformer to the reference mol.
        shape_sim (float): The shape similarity score.
        esp_sim (float): The esp similarity score.
        total_sim (float): The total (combined) score.
    """
    # Generate conformers for the query smiles
    conf_scores, conf_ids, query_confs = get_confs_constant_seed(query_smi, ref_mol)
    
    # Early stop if no conformers were generated
    if len(conf_scores) == 0:
        print("Conformer generation failed for:")
        print(query_smi)
        return None, None, None, None

    # Create a new mol for the best scoring conformation
    best_conf_id = int(np.argmax(conf_scores))
    best_conf = isolate_conformer(query_confs.mol, best_conf_id)

    # Calculate espsim similarity
    shape_sim = GetShapeSim(query_confs.mol, ref_mol, prbCid=best_conf_id, refCid=0)
    esp_sim = GetEspSim(query_confs.mol, ref_mol, prbCid=best_conf_id, refCid=0)
    total_sim = shape_sim + esp_sim

    return best_conf, shape_sim, esp_sim, total_sim

def generate_espsim_df(df:pd.DataFrame, smiles_col:str, ref_mol):
    """
    Calculates and collects espsim score data (shape and ESP similarity) for smiles in dataframe with a reference mol.

    Parameters:
        df (DataFrame): DataFrame containing the smiles for scoring.
        smiles_col (str): The name of the dataframe containing the smiles to use for .
        ref_mol (Mol): Reference molecule (RDKit Mol object).

    Returns:
        espsim df (DataFrame): A dataframe containing conformer mols, shape similarity scores, esp similarity scores, and total similarity scores.
    """
    sim_data = []
    for smi in tqdm(df[smiles_col]):
        best_conf, shape_sim, esp_sim, total_sim = get_esp_similarity_score(smi, ref_mol)
        if best_conf == None:
            continue
        sim_series = pd.Series({'conformer_mol':best_conf, 'shape_sim':shape_sim,  'esp_sim': esp_sim, 'total_sim':total_sim})
        sim_data.append(sim_series)
    espsim_df = pd.concat(sim_data, axis=1).T.reset_index(drop=True)
    return espsim_df



def draw_overlays(ref_mol, df, width=1000, height=2000, grid_size=(6, 3)):
    """
    Draws the superimposed 3D structures of a probe molecule and a reference molecule using py3Dmol.

    Parameters:
        ref_mol (Mol): Reference molecule (RDKit Mol object).
        df (DataFrame): DataFrame containing probe molecules. Assumes a column 'conformer_mol' with RDKit Mol objects.
        width (int): Width of the py3Dmol viewer.
        height (int): Height of the py3Dmol viewer.
        grid_size (tuple): Grid size for displaying molecules.

    Returns:
        py3Dmol.view: The py3Dmol viewer with the superimposed molecules displayed.
    """
    view = py3Dmol.view(width=width, height=height, linked=False, viewergrid=grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            mol_position_on_df = i * grid_size[1] + j

            # Check if the index is within the bounds of the DataFrame
            if mol_position_on_df >= len(df):
                continue

            # Extract the conformer molecule
            prbMol = deepcopy(df.iloc[mol_position_on_df]['conformer_mol'])

            # Add the probe and reference molecules to the viewer
            view.addModel(Chem.MolToMolBlock(prbMol), 'mol', viewer=(i, j))
            view.addModel(Chem.MolToMolBlock(ref_mol), 'mol', viewer=(i, j))

            # Set styles for the viewer
            view.setStyle({'stick': {}}, viewer=(i, j))
            view.setStyle({'model': 0}, {'stick': {'colorscheme': 'greenCarbon'}}, viewer=(i, j))

    view.zoomTo()
    return view.render()

def draw_overlay(ref_mol, mol, width=400, height=400):
    """
    Draws the superimposed 3D structures of a probe molecule and a reference molecule using py3Dmol.

    Parameters:
        ref_mol (Mol): Reference molecule (RDKit Mol object).
        df (DataFrame): DataFrame containing probe molecules. Assumes a column 'conformer_mol' with RDKit Mol objects.
        width (int): Width of the py3Dmol viewer.
        height (int): Height of the py3Dmol viewer.
        grid_size (tuple): Grid size for displaying molecules.

    Returns:
        py3Dmol.view: The py3Dmol viewer with the superimposed molecules displayed.
    """
    view = py3Dmol.view(width=width, height=height, linked=False)


    # Add the probe and reference molecules to the viewer
    view.addModel(Chem.MolToMolBlock(mol), 'mol')
    view.addModel(Chem.MolToMolBlock(ref_mol), 'mol')

    # Set styles for the viewer
    view.setStyle({'stick': {}})
    view.setStyle({'model': 0}, {'stick': {'colorscheme': 'greenCarbon'}})

    view.zoomTo()
    return view.render()