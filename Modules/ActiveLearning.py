import pandas as pd
import numpy as np
from Reactions import *
from ESPsim import*

from rdkit.Chem import AllChem
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class Conformer:
    '''A class for generating molecule conformers.'''
    def __init__(self, smiles, n_confs, seed = 0xf00d):
        # Define the conformer generator
        p = AllChem.ETKDGv3()
        if seed == 'random':
            p.randomSeed = 0xf00a
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

def assemble_three_cycle_BBs(linker_smi:str, CA_smi:str, Amine_smi:str):
    int_product_smi = ''
    product_smi = ''
    run_reaction = Reactions()
    linker = Reactant(linker_smi, 'aryl_halide', 'aryl_amine')
    carb_acid = Reactant(CA_smi, 'carboxylic_acid')
    amine= Reactant(Amine_smi, 'alkyl_amine', 'aryl_amine')
    
    try:
        int_product_smi = (run_reaction.amide_bond_formation(linker, carb_acid))
    except IndexError:
        return np.nan  
    
    int_linker = Reactant(int_product_smi, 'aryl_halide')

    try:
        product_smi= run_reaction.buchwald_hartwig_coupling(int_linker, amine)
    except IndexError:
        return np.nan

    return product_smi


def create_morgan_fingerprints(df: pd.DataFrame, smiles_col:str, mol_col:str = None) -> pd.DataFrame:
    """Creates Morgan fingerprints for the input library.

    Args:
        df (pd.DataFrame): The input dataframe containing the smiles to fingerprint.
        smiles (str): The name of the smiles column.
        mol

    Returns:
        pd.DataFrame: The Morgan fingerprints of the input library.
    """
    #Add mols in no mol_col is provided 
    if mol_col == None:
        mol_col = 'ROMol'
        print('Generating Mols...')
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smiles_col, molCol=mol_col)

    #Generate the FPs
    fps = [list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024 )) for mol in tqdm(df[mol_col]) ]
    
    #Create a dataframe for the FP data
    fps_df = pd.DataFrame(fps, columns=[f"fp_{x}" for x in range(len(fps[0]))], index=df.index)
        
    return fps_df


def train_ml_model(
    library_df: pd.DataFrame, fingerprint_df: pd.DataFrame
) -> RandomForestRegressor:
    """
    Trains a random forest regressor model on scoring function values from the input library.

    Args:
        library_df (pd.DataFrame): The input library containing "sf_scores" column 
                                   for values from a scoring function.
        fingerprint_df (pd.DataFrame): Morgan fingerprints of the input library.
                                       Must share the same index as `library_df`.

    Returns:
        RandomForestRegressor: The trained random forest regressor model.
    """
    # Filter rows with non-NaN "sf_scores"
    scored_df = library_df.dropna(subset=["sf_scores"])

    # Extract features and target values
    X = fingerprint_df.loc[scored_df.index]
    y = scored_df["sf_scores"]

    # Initialize and train the random forest regressor
    regressor = RandomForestRegressor(max_depth=10, random_state=42)
    regressor.fit(X, y)

    return regressor


def score_library(
    library_df: pd.DataFrame,
    regressor: RandomForestRegressor,
    fingerprint_df: pd.DataFrame,
) -> pd.DataFrame:
    """Scores the entire library with the trained model.

    Args:
        library_df (pd.DataFrame): The input library.
        regressor (RandomForestRegressor): The trained random forest regressor model.
        fingerprints (pd.DataFrame): The Morgan fingerprints of the input library.

    Returns:
        pd.DataFrame: The input library with the model scores.
    """
    library_df["model_scores"] = regressor.predict(fingerprint_df)
    return library_df


def espsim_scoring_function(smiles_list, ref_mol):
    score_list = []
    for smiles in tqdm(smiles_list):
        best_conf, shape_sim, esp_sim, total_sim = get_esp_similarity_score(smiles, ref_mol)
        score_list.append(total_sim)
    return score_list

def run_active_learning_with_metrics(
    library_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    BB_Cols:list,
    compounds_per_round: int,
    number_of_rounds: int,
    scoring_function: callable,
    minimize: bool,
    ref_mol: Chem.Mol,
    early_stopping_value: float | None = None,
) -> pd.DataFrame:
    
    """Runs active learning on the virtual library using a recursively applied scoring function."""
    # Select initial random sample
    sample_df = library_df.sample(compounds_per_round)
    library_df["sf_scores"] = np.NaN


    ################################################
    #Build the libary cpds for the initial sample 
    linker_col = BB_Cols[0]
    acid_col = BB_Cols[1]
    amine_col = BB_Cols[2]

    linkers = list(sample_df[linker_col])
    acids = list(sample_df[acid_col])
    amines = list(sample_df[amine_col])
    prods = [assemble_three_cycle_BBs(linkers[i], acids[i], amines[i]) for i in range(len(linkers))]
    sample_df['Assembled_Smiles'] = prods
    #sample_df['Assembled_Smiles'] = sample_df.apply(lambda row: assemble_BBs(row[linker_col], row[acid_col], row[amine_col]))
    initial_sample = sample_df.dropna(subset=['Assembled_Smiles']) #removes any rows with reactions that may have failed
    ################################################

    # Score the initial sample using the smiles from the assembled BBs
    initial_scores = scoring_function(initial_sample['Assembled_Smiles'].to_list(), ref_mol)

    # Save the scoring function scores in the library_df
    library_df.loc[initial_sample.index, "sf_scores"] = initial_scores

    # Track metrics for each round
    metrics_dict = {'round':[], 'spearman':[], 'mae':[]}
    # Run active learning
    al_round = 0
    while al_round < number_of_rounds:
        # Train the ML model
        model = train_ml_model(library_df, fingerprint_df)

        # Use the model to score the entire virtual library
        library_df = score_library(library_df, model, fingerprint_df)

        # Evaluate metrics for the current round
        scored_library = library_df[library_df["sf_scores"].notna()]
        model_predictions = scored_library["model_scores"]  # Running `score_library` updates this column
        true_scores = scored_library["sf_scores"]

        if not model_predictions.empty and not true_scores.empty:
            # Calculate Spearman's rank correlation
            spearman_corr = spearmanr(model_predictions, true_scores).correlation

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(true_scores, model_predictions)


            metrics_dict["round"].append(al_round)
            metrics_dict["spearman"].append(spearman_corr)
            metrics_dict["mae"].append(mae)

            print(f"Round {al_round} ({len(scored_library) } compounds): Spearman Correlation = {spearman_corr:.4f}, MAE = {mae:.4f}")

        # Select the top scoring molecules that havent been scored by the scoring function
        top_compounds = (
            library_df[library_df["sf_scores"].isna()]
            .sort_values("model_scores", ascending=minimize)  # Use `model_scores` for selection
            .head(compounds_per_round)
        )


        top_linkers = list(top_compounds[linker_col])
        top_acids = list(top_compounds[acid_col])
        top_amines = list(top_compounds[amine_col])
        top_prods = [assemble_three_cycle_BBs(top_linkers[i], top_acids[i], top_amines[i]) for i in range(len(top_linkers))]
        top_compounds['Assembled_Smiles'] = top_prods
        #sample_df['Assembled_Smiles'] = sample_df.apply(lambda row: assemble_BBs(row[linker_col], row[acid_col], row[amine_col]))
        update_df = top_compounds.dropna(subset=['Assembled_Smiles']) #removes any rows with reactions that may have failed

        # Score the top molecules with the slow function
        slow_scores = scoring_function(update_df['Assembled_Smiles'].to_list(), ref_mol)

        # Save the slow scores
        library_df.loc[update_df.index, "sf_scores"] = slow_scores
        library_df.loc[update_df.index, "scored_round"] = al_round

        al_round += 1
        if early_stopping_value is not None:
            if minimize:
                if library_df["sf_scores"].min() <= early_stopping_value:
                    break
            else:
                if library_df["sf_scores"].max() >= early_stopping_value:
                    break

    # Save metrics information to the library (optional)
    #library["metrics_per_round"] = metrics_per_round

    return library_df, metrics_dict



def embed_with_timeout(mol, n_confs, params, timeout=10):
    """
    Wrapper for AllChem.EmbedMultipleConfs with a timeout.

    Parameters:
        mol (rdkit.Chem.Mol): The RDKit molecule.
        n_confs (int): Number of conformers to generate.
        params (AllChem.ETKDG): Parameters for embedding.
        timeout (int): Timeout in seconds.

    Returns:
        list: List of conformer IDs, or None if the timeout occurs.
    """
    def embed():
        return AllChem.EmbedMultipleConfs(mol, n_confs, params)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(embed)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Embedding conformers timed out after {timeout} seconds.")
            return None

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
    query_confs = Conformer(query_smi, 100, seed = 'random')
    if query_confs.conformer_ids == None:
        return [], [], []
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
    conf_scores, conf_ids, query_confs = get_confs_random_seed(query_smi, ref_mol)
    
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
