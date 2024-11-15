# Standard library imports
import os
from itertools import  chain

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import mols2grid

# RDKit and cheminformatics
from rdkit import Chem 
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import Descriptors,rdFMCS,rdMolDescriptors, rdMolAlign
from rdkit.Chem.Draw import IPythonConsole

# Useful RDKit utilities
import useful_rdkit_utils as uru

# Progress bar
from tqdm import tqdm

# Configure environment and settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IPythonConsole.ipython_3d = True
tqdm.pandas()



#########################################
############    Functions    ############
#########################################

def dataframe_from_smiles(smiles_list: list, add_properties: bool = True) -> pd.DataFrame:
    """
    Creates a DataFrame from a list of SMILES strings, calculates molecular properties
    (optionally), and removes duplicate entries based on InChIKey.

    Parameters:
        smiles_list (list): List of SMILES strings.
        add_properties (bool): If True, calculates molecular properties for each molecule.

    Returns:
        pd.DataFrame: A DataFrame containing SMILES, molecular properties, and deduplicated entries.
    """
    # Create a DataFrame with the SMILES column
    df = pd.DataFrame({'smiles': smiles_list})

    print(f"Starting with {len(df)} compounds...")
    
    # Add the RDKit molecule column
    print('Generating Mols...')
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='Mol')

    # Drop rows where molecule generation failed
    df = df[df['Mol'].notnull()]
    print(f"Valid molecules: {len(df)}")

    if add_properties:
        #tqdm.pandas(desc="Calculating Descriptors")
        try:
            # Calculate molecular properties
            print('Calculating molecular weights...')
            df['mol_wt'] = df['Mol'].progress_apply(Descriptors.ExactMolWt)
            print('Calculating LogP values...')
            df['logp'] = df['Mol'].progress_apply(Descriptors.MolLogP)
            print('Calculating H-bond donors...')
            df['h_donors'] = df['Mol'].progress_apply(Descriptors.NumHDonors)
            print('Calculating H-bond acceptors...')
            df['h_acceptors'] = df['Mol'].progress_apply(Descriptors.NumHAcceptors)
            print('Calculating rotatable bonds...')
            df['rotatable_bonds'] = df['Mol'].progress_apply(Descriptors.NumRotatableBonds)
            print('Calculating polar surface area...')
            df['polar_surface_area'] = df['Mol'].progress_apply(lambda mol: Chem.QED.properties(mol).PSA)
            print('Calculating number of rings...')
            df['rings'] = df['Mol'].progress_apply(rdMolDescriptors.CalcNumRings)
            print('Calculating number of atoms...')
            df['atoms'] = df['Mol'].progress_apply(lambda mol: mol.GetNumAtoms())
            print('Calculating number of heavy atoms...')
            df['heavy_atoms'] = df['Mol'].progress_apply(lambda mol: mol.GetNumHeavyAtoms())
            
        except Exception as e:
            print(f"Error during property calculation: {e}")

    # Calculate InChIKey for deduplication
    print('Calculating InChIKeys...')
    df['inchi'] = df['Mol'].progress_apply(Chem.MolToInchiKey)

    # Remove duplicates based on InChIKey
    initial_count = len(df)
    df.drop_duplicates(subset='inchi', inplace=True)
    print('Removing duplicates...')
    print(f"Before: {initial_count} | After: {len(df)}")

    return df



def REOS_filter(df: pd.DataFrame, mol_col: str, apply_filter: bool = False) -> pd.DataFrame:
    """
    Applies the REOS (Rapid Elimination of Swill) filter to a DataFrame of molecules and optionally filters the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing molecular data.
        mol_col (str): The column name in the DataFrame containing RDKit molecule objects.
        apply_filter (bool): If True, filters the DataFrame to keep only molecules passing the REOS filter.

    Returns:
        pd.DataFrame: 
            - The updated or filtered DataFrame with REOS results.
            - A DataFrame with a summary of the REOS filter results.
    """
    # Initialize the REOS object and set active rule sets
    reos = uru.REOS()
    reos.set_active_rule_sets(["Dundee"])

    # Apply the REOS filter to the molecules
    df[['rule_set', 'reos']] = df[mol_col].apply(lambda x: pd.Series(reos.process_mol(x)))

    # Display value counts for REOS results
    reos_df = uru.value_counts_df(df, "reos")
    
    # Optionally filter the DataFrame to include only molecules passing the REOS filter
    before_count=(len(df))
    if apply_filter:
        df = df[df['reos'] == 'ok']
        print(f'Removed {before_count - len(df)} REOS violations.')
    return df, reos_df



def lipinski_verber_filter_full(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Filters a DataFrame of molecules based on Lipinski and Veber rules for druglikeness.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing molecular properties.
        strict (bool): If True, applies stricter Lipinski and Veber thresholds.

    Returns:
        pd.DataFrame: The filtered DataFrame with molecules passing all druglikeness rules.
    """
    before_count = len(df)

    # Required columns for Lipinski filtering
    required_columns = ['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'polar_surface_area']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The DataFrame is missing the following required columns: {', '.join(missing_columns)}")
    
    # Define the filter criteria
    criteria = {
        'mol_wt': 500 if strict else 600,
        'rot_bonds': 5 if strict else 10,
        'logP': 5,
        'h_donors': 3,
        'h_acceptors': 10,
        'polar_surface_area': 140
    }

    # Apply the filter
    filtered_df = df[
        (df['mol_wt'] <= criteria['mol_wt']) &
        (df['rotatable_bonds'] <= criteria['rotatable_bonds']) &
        (df['logp'] <= criteria['logp']) &
        (df['h_donors'] <= criteria['h_donors']) &
        (df['h_acceptors'] <= criteria['h_acceptors']) &
        (df['polar_surface_area'] <= criteria['polar_surface_area'])
    ]

    after_count = len(filtered_df)
    print(f"Removed {before_count - after_count} druglikeness violations.")
    
    return filtered_df


    
def lipinski_verber_filter_flexible(df: pd.DataFrame, mol_col: str, filter_threshold: int = 6) -> pd.DataFrame:
    """
    Filters a DataFrame of molecules based on Lipinski's Rule of Five criteria.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing molecular properties.
        filter_threshold (int): The minimum number of Lipinski criteria a molecule must pass to be retained.

    Returns:
        pd.DataFrame: A DataFrame containing molecules that meet the specified filter threshold.
    """
    # Required columns for Lipinski filtering
    required_columns = ['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'polar_surface_area']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The DataFrame is missing the following required columns: {', '.join(missing_columns)}")

    # Initialize a dictionary to track how many molecules pass each criterion
    lipinski_params = {col: 0 for col in required_columns}
    lipinski_params['All'] = 0

    # Define the Lipinski criteria as lambda functions
    criteria = {
        'mol_wt': lambda row: row['mol_wt'] <= 500,
        'logp': lambda row: row['logp'] <= 5,
        'h_donors': lambda row: row['h_donors'] <= 5,
        'h_acceptors': lambda row: row['h_acceptors'] <= 10,
        'rotatable_bonds': lambda row: row['rotatable_bonds'] <= 5,
        'polar_surface_area': lambda row: row['polar_surface_area'] <= 140
    }

    # Count how many criteria each molecule passes
    df['lipinski_count'] = df.apply(
        lambda row: sum(1 for key, func in criteria.items() if func(row)), axis=1
    )

    # Update the Lipinski parameters tracker
    for key, func in criteria.items():
        lipinski_params[key] = df.apply(func, axis=1).sum()
    lipinski_params['All'] = (df['lipinski_count'] == len(criteria)).sum()

    # Filter the DataFrame based on the threshold
    filtered_df = df[df['lipinski_count'] >= filter_threshold]

    print(f"Lipinski Filter Summary: {lipinski_params}")
    print(f"Total molecules before filter: {len(df)} | After filter: {len(filtered_df)}")
    
    return filtered_df



def calculate_properties(df: pd.DataFrame, mol_col: str) -> pd.DataFrame:
    """
    Calculates molecular property descriptors for RDKit molecules in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecules.
        mol_col (str): Column name containing RDKit molecule objects.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated molecular properties.
    """
    print('Calculating molecular property descriptors...')
    
    # Define the property calculations
    df['mol_wt'] = df[mol_col].apply(Descriptors.ExactMolWt)
    df['logp'] = df[mol_col].apply(Descriptors.MolLogP)
    df['h_donors'] = df[mol_col].apply(Descriptors.NumHDonors)
    df['h_acceptors'] = df[mol_col].apply(Descriptors.NumHAcceptors)
    df['rotatable_bonds'] = df[mol_col].apply(Descriptors.NumRotatableBonds)
    df['polar_surface_area'] = df[mol_col].apply(lambda mol: Chem.QED.properties(mol).PSA)
    df['atoms'] = df[mol_col].apply(lambda mol: mol.GetNumAtoms())
    df['heavy_atoms'] = df[mol_col].apply(lambda mol: mol.GetNumHeavyAtoms())
    df['rings'] = df[mol_col].apply(Chem.rdMolDescriptors.CalcNumRings)
    
    return df



def analyze_ring_systems(df: pd.DataFrame, mol_col: str):
    """
    Analyzes ring systems in molecules from a DataFrame, calculates their frequencies,
    and displays the results using mols2grid.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecular data.
        mol_col (str): Column name containing RDKit Mol objects.

    Returns:
        mols2grid.MolsGrid: A MolsGrid display object showing the ring systems and their counts.
    """
    # Ensure the DataFrame is not modified in-place
    df = df.copy()

    # Initialize the RingSystemFinder
    ring_system_finder = uru.RingSystemFinder()

    # Find ring systems for each molecule and store them in a new column
    df['ring_systems'] = df[mol_col].apply(ring_system_finder.find_ring_systems)

    # Flatten the list of ring systems across all molecules
    ring_list = chain.from_iterable(df['ring_systems'])

    # Convert the ring systems into a Series and count occurrences
    ring_series = pd.Series(ring_list)
    ring_counts = ring_series.value_counts()

    # Create a DataFrame for the ring systems and their counts
    ring_df = ring_counts.reset_index()
    ring_df.columns = ["SMILES", "Count"]

    # Display the results using mols2grid
    return mols2grid.display(ring_df, smiles_col="SMILES", subset=["img", "Count"], selection=False)



def display_molgrid(df: pd.DataFrame, smi_col: str):
    """
    Displays a molecule grid using mols2grid.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecular data.
        smi_col (str): Column name containing SMILES strings.

    Returns:
        mols2grid.MolsGrid: A MolsGrid display object.
    """
    return mols2grid.display(
        df,
        smiles_col=smi_col,
        selection=False,
        size=(300, 150))


def drop_duplicate_smiles(df:pd.DataFrame, smi_col = "Smiles"):
    '''Drop rows with duplicate smiles from a dataframe using the specified column'''
    before_rows = df.shape[0]
    df.drop_duplicates(smi_col,inplace=True)
    after_rows = df.shape[0]
    print(f"{before_rows} rows reduced to {after_rows} rows")



def align_mols_to_template(df: pd.DataFrame, mol_col: str, template_mol: Chem.Mol) -> pd.DataFrame:
    """
    Aligns molecules in a DataFrame column to a template molecule.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecule data.
        mol_col (str): Column name containing RDKit Mol objects.
        template_mol (Chem.Mol): RDKit Mol object to use as the alignment template.

    Returns:
        pd.DataFrame: Updated DataFrame with aligned molecules in the specified column.
    """
    # Validate input parameters
    if mol_col not in df.columns:
        raise ValueError(f"Column '{mol_col}' not found in the DataFrame.")
    if not isinstance(template_mol, Chem.Mol):
        raise ValueError("The 'template_mol' parameter must be an RDKit Mol object.")

    print("Aligning molecules to the template...")
    aligned_mols = []

    for mol in df[mol_col]:
        try:
            # Align molecule to the template
            aligned_mol = AllChem.GenerateDepictionMatching2DStructure(mol, template_mol)
        except ValueError as e:
            print(f"Alignment failed for a molecule: {e}. Keeping the original molecule.")
            aligned_mol = mol  # Retain the original molecule if alignment fails
        aligned_mols.append(aligned_mol)

    # Update the DataFrame with aligned molecules
    df = df.copy()  # Avoid modifying the original DataFrame
    df[mol_col] = aligned_mols

    return df


def add_images(df: pd.DataFrame, mol_col: str, size: tuple = (150, 150)) -> pd.DataFrame:
    """
    Generates 2D molecule images (SVG format) for molecules in a DataFrame column and adds them to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecule data.
        mol_col (str): Column name containing RDKit Mol objects.
        size (tuple): Tuple specifying the width and height of the generated images (default: (150, 150)).

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'imgs' column containing SVG images of molecules.
    """
    # Validate inputs
    if mol_col not in df.columns:
        raise ValueError(f"Column '{mol_col}' not found in the DataFrame.")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("The 'size' parameter must be a tuple of length 2 (width, height).")

    print("Generating molecule images...")
    imgs = []

    for mol in tqdm(df[mol_col], desc="Rendering molecules"):
        if not isinstance(mol, Chem.Mol):
            imgs.append(None)  # Handle non-molecule entries gracefully
            continue
        try:
            d2d = Draw.MolDraw2DSVG(size[0], size[1])
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            svg = d2d.GetDrawingText()
            imgs.append(svg)
        except Exception as e:
            print(f"Failed to generate image for a molecule: {e}")
            imgs.append(None)

    # Add the images to a new column in the DataFrame
    df = df.copy()  # Avoid modifying the original DataFrame
    df['imgs'] = imgs

    return df


def plot_df(df:pd.DataFrame):
    '''Plot the columns in a dataframe as histograms. The dataframe must contain only numerical or catagorical data.
    
    Remove any columns containing non-numerical or non-catagorical data.'''

    # Number of columns in the DataFrame
    num_columns = len(df.columns)

    # Set up the subplot grid with an appropriate number of rows and columns
    fig, axes = plt.subplots(nrows=(num_columns + 2) // 3, ncols=3, figsize=(12, num_columns * 2))
    axes = axes.flatten()  # Flatten in case we need a 1D list of axes for easy iteration

    for i, column in enumerate(df.columns):
        df[column].plot(kind='hist', ax=axes[i], title=column)
        axes[i].set_ylabel("Count")
        axes[i].set_xlabel(column)

    # Hide any extra subplots if the grid is larger than the number of columns
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()





