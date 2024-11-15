# Standard library imports
import os
from itertools import product, combinations, chain
from typing import Tuple, Optional, List

# Data manipulation and analysis
import pandas as pd
import numpy as np


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Viridis256, Turbo256, Inferno256
import py3Dmol
import mols2grid

# Machine learning and clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap

# RDKit and cheminformatics
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import Descriptors,rdFMCS,rdMolDescriptors,rdMolAlign
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.Features.ShowFeats import _featColors as featColors
from rdkit.Chem.rdFMCS import FindMCS
import datamol as dm

# MHFP and molecular fingerprints
from mhfp.encoder import MHFPEncoder
from mxfp import mxfp

# Useful RDKit utilities
import useful_rdkit_utils as uru

# Progress bar
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

# Configure environment and settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IPythonConsole.ipython_3d = True
tqdm.pandas()



#########################################
############     Classes     ############
#########################################


class Properties:
    """
    A class to calculate and store molecular properties for a list of SMILES strings in a DataFrame.

    Attributes:
        data_df (pd.DataFrame): A DataFrame containing SMILES, IDs, and RDKit Mol objects.
        property_dict (dict): A dictionary to store calculated molecular properties.

    Parameters:
        smiles (list[str]): A list of SMILES strings to calculate properties for.
        ids (list[str]): A list of corresponding identifier labels for the SMILES strings.
    """
    def __init__(self, smiles: list[str], ids: list[str]):
        if len(smiles) != len(ids):
            raise ValueError("The length of 'smiles' and 'ids' must be the same.")

        self.data_df = pd.DataFrame({'ID': ids, 'Smiles': smiles})

        print('Generating molecules...')
        tqdm.pandas(desc="Processing SMILES")
        # Generate RDKit Mol objects
        self.data_df['ROMol'] = self.data_df['Smiles'].progress_apply(Chem.MolFromSmiles)

        # Convert back to SMILES (standardized format)
        self.data_df['Smiles'] = self.data_df['ROMol'].apply(
            lambda mol: Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False) if mol else None
        )

        # Initialize property dictionary
        self.property_dict = {
            'ID': self.data_df['ID'].tolist(),
            'smiles': self.data_df['Smiles'].tolist(),
            'mol_wt': [],
            'logp': [],
            'h_donors': [],
            'h_acceptors': [],
            'rotatable_bonds': [],
            'polar_surface_area': [],
            'atoms': [],
            'heavy_atoms': [],
            'rings': []
        }
        
        #Populate the property dict mol by mol
        print('Calculating molecular property descriptors...')
        for mol in tqdm(self.data_df['ROMol']):
            self.property_dict['mol_wt'].append(Descriptors.ExactMolWt(mol))
            self.property_dict['logp'].append(Descriptors.MolLogP(mol))
            self.property_dict['h_donors'].append(Descriptors.NumHDonors(mol))
            self.property_dict['h_acceptors'].append(Descriptors.NumHAcceptors(mol))
            self.property_dict['rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
            self.property_dict['polar_surface_area'].append(Chem.QED.properties(mol).PSA)
            self.property_dict['atoms'].append(Chem.rdchem.Mol.GetNumAtoms(mol))
            self.property_dict['heavy_atoms'].append(Chem.rdchem.Mol.GetNumHeavyAtoms(mol))
            self.property_dict['rings'].append(Chem.rdMolDescriptors.CalcNumRings(mol))

        self.property_df=pd.DataFrame(self.property_dict)
        self.property_df.to_csv('Properties.csv')

    def apply_lipinski_filter(self):
        Lipinski_counter=0
        Lipinski_params={'mol_wt':0, 'logp':0 ,'h_donors':0,'h_acceptors':0,'rotatable_bonds':0,'polar_surface_area':0, 'All':0}
        n_passing=[]
        passing_smiles=[]
        druglike_df=pd.DataFrame()
        for index, row in self.property_df.iterrows():
            Lipinski_counter=0
            if row['mol_wt'] <= 500:
                Lipinski_params['mol_wt']+=1
                Lipinski_counter+=1
            if row['logp']<= 5:
                Lipinski_params['logp']+=1
                Lipinski_counter+=1
            if row['h_donors'] <= 5:
                Lipinski_params['h_donors']+=1
                Lipinski_counter+=1
            if row['h_acceptors'] <= 10:
                Lipinski_params['h_acceptors']+=1
                Lipinski_counter+=1
            if row['rotatable_bonds'] <= 5:
                Lipinski_params['rotatable_bonds']+=1
                Lipinski_counter+=1
            if row['polar_surface_area'] <=140:
                Lipinski_params['polar_surface_area']+=1
                Lipinski_counter+=1
            if Lipinski_counter==6:
                Lipinski_params['All']+=1
            n_passing.append(Lipinski_counter)

        self.property_df['lipinski_count'] = n_passing
        druglike_df = self.property_df[self.property_df['lipinski_count'] == 6]
        return(Lipinski_params,druglike_df)
    
    def assemble_plot_df(self, df:pd.DataFrame, property:str):
        plot_df=pd.DataFrame()
        property_vals=np.array(df[property].values.tolist())
        plot_df[property] = property_vals
        return(plot_df)
    
    def plot_kde(self, df:pd.DataFrame, col:str):
        '''Generate a KDE plot of the data from a single column in a dataframe.'''
        data= df
        sns.kdeplot(data=data, x=col)
        
    def Property_Dataframe(self,property): 
        '''Generate a dataframe for a specfic property column in an input csv file'''
        self.prop_df=pd.DataFrame() #create an empty df for holding the properties
        Property=[]
        #For each type in the csv "DEL" column:
        for Del in self.DEL_list:
            DEL_df=pd.DataFrame()
            temp_df=self.data_df.loc[self.data_df['DEL'] == str(Del)] #create a temp dataframe for the current 'Del' iteration
            temp_properties=np.array(temp_df[str(property)].values.tolist())  #extract the property values as np array
            DEL_df[Del]=temp_properties #create the DEL_df for the current Del iteration
            self.prop_df=pd.concat([self.prop_df, DEL_df], axis=1) #Concatonate the DEL_df to the main prop_df
        return(self.prop_df)


class Fingerprint:
    """
    Generates molecular fingerprints for a list of SMILES using the specified descriptor and bit size.

    Attributes:
        fp_df (pd.DataFrame): DataFrame containing the SMILES, IDs, and molecular data.
        fp_array (np.ndarray): NumPy array containing the generated fingerprints.

    Parameters:
        smiles (list[str]): A list of SMILES strings.
        ids (list[str]): A list of identifier labels for the SMILES strings.
        measure (list[float]): A list of binding measurements corresponding to the SMILES.
        descriptor (str): The fingerprinting method to use. Valid methods are: "MXFP", "MHFP6", "ECFP6".
        nBits (int): The number of bits to use for fingerprinting (applicable for some descriptors).
    """
    def __init__(self, smiles:list, ids:list, measure:list, descriptor:str, nBits:int):
        self.nBits=nBits
        self.descriptor=descriptor
        self.fp_df = pd.DataFrame() #Create df from smiles list
        self.fp_df['Smiles']= smiles
        self.fp_df['ID']= ids
        self.fp_df['Binding']= measure
        self.fp_df['ROMol'] = self.fp_df.Smiles.apply(Chem.MolFromSmiles) #Create mols from smiles
        self.fp_df['Smiles'] = self.fp_df.ROMol.apply(lambda x: Chem.MolToSmiles(x, kekuleSmiles=True, isomericSmiles=False)) #Cleanup the smiles
        
        if self.descriptor=='MXFP':
            MXFP = mxfp.MXFPCalculator(dimensionality='2D')
            fp = [MXFP.mxfp_from_mol(x) for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            self.fp_array=np.array(fp)
            self.nBits=217

        if self.descriptor=='MHFP6':
            MHFP6 = MHFPEncoder(n_permutations=self.nBits)
            fp=[MHFP6.encode(x) for x in tqdm(self.fp_df['Smiles'],desc="Fingerprinting")]
            self.fp_array=np.array(fp)

        if self.descriptor=='ECFP6':
            ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x,radius=3, nBits=self.nBits) for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            fp_list=[list(l) for l in ECFP6]
            self.fp_array = np.array(fp_list)
        
        if self.descriptor=='ECFP4':
            ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x,radius=2, nBits=self.nBits) for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            fp_list=[list(l) for l in ECFP6]
            self.fp_array = np.array(fp_list)

        if self.descriptor=='MACCS':
            fp_list=[]
            MACCS=[rdMolDescriptors.GetMACCSKeysFingerprint(x) for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            fp_str=[fp.ToBitString() for fp in MACCS]
            for fp in fp_str:
                temp_list=[int(x) for x in fp]
                fp_list.append(temp_list)
            self.fp_array = np.array(fp_list)
            self.nBits=167

        if self.descriptor=='MQN':
            fp_list=[]
            MQN=[rdMolDescriptors.MQNs_(x)  for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            self.fp_array=np.array(MQN)
            self.nBits=42

        if self.descriptor=='AP':
            AP=[DataStructs.cDataStructs.FoldFingerprint(Pairs.GetAtomPairFingerprintAsBitVect(x),foldFactor=4096) for x in tqdm(self.fp_df['ROMol'],desc="Fingerprinting")]
            self.fp_array=np.array(AP)
            self.nBits=2048

        #add the fps to the df and then preview 
        self.fp_df['fp'] = list(self.fp_array)
        self.fp_df.head()

class Similarity(Fingerprint):
    """
    A class for calculating fingerprint similarities and/or distance metrics.

    Attributes:
        descriptor (str): The fingerprint descriptor used.
        nBits (int): Number of bits in the fingerprint.
        fp_df (pd.DataFrame): DataFrame containing fingerprints and associated data.
        metric (str): The distance metric to apply.

    Parameters:
        fp (Fingerprint): An instance of the Fingerprint class.
        metric (str): A distance metric from sklearn. Valid options include: "cityblock", "cosine", "euclidean", 
                      "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard".
    """

    def __init__(self, fp: Fingerprint, metric: str):
        if metric not in {
            "cityblock", "cosine", "euclidean", "manhattan", "braycurtis",
            "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard"
        }:
            raise ValueError(f"Invalid metric: {metric}. Please choose a valid metric from sklearn.")
        
        self.descriptor = fp.descriptor
        self.nBits = fp.nBits
        self.fp_df = fp.fp_df
        self.metric = metric  

    def pairwise_dist_from_subtypes(self, 
        df: pd.DataFrame, 
        grouping_col: str, 
        coord_cols: Tuple[str, str], 
        metric: Optional[str] = None, 
        n_samples: Optional[int] = None) -> pd.DataFrame:
    
        """
        Calculate pairwise distances for subtype groupings in an input DataFrame with coordinate columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing a grouping column and a set of coordinate columns.
            grouping_col (str): The name of the column used for grouping (e.g., "cluster", "target").
            coord_cols (tuple): Names of the two columns with coordinate data (e.g., ("x1", "x2")).
            metric (str, optional): Distance metric to use with sklearn. Defaults to the metric set in the class.
            n_samples (int, optional): Number of subsamples to take from each group. Defaults to None (use all data).

        Returns:
            pd.DataFrame: A DataFrame containing the pairwise distances for each item, grouped by the specified column.
        """
        metric = metric or self.metric

        # Validate input
        if grouping_col not in df.columns:
            raise ValueError(f"Column '{grouping_col}' not found in the DataFrame.")
        if any(col not in df.columns for col in coord_cols):
            raise ValueError(f"Coordinate columns {coord_cols} not found in the DataFrame.")

        print(f"Calculating pairwise distances using {metric} metric...")
        dist_df = pd.DataFrame()

        for group in df[grouping_col].unique():
            # Subset and sample data
            group_df = df[df[grouping_col] == group]
            if n_samples is not None and n_samples < len(group_df):
                group_df = group_df.sample(n_samples)

            # Fill missing values in coordinate columns
            group_df[coord_cols] = group_df[coord_cols].fillna(0)

            # Prepare coordinate array
            xy_array = group_df[list(coord_cols)].to_numpy()

            # Compute pairwise distances
            distances = pairwise_distances(xy_array, xy_array, metric=metric)

            # Flatten and pad distances to align with DataFrame structure
            distances_flat = distances.flatten()
            max_len = max(len(distances_flat), len(dist_df))
            padded_distances = np.pad(distances_flat, (0, max_len - len(distances_flat)), constant_values=np.nan)

            dist_df[group] = pd.Series(padded_distances)

        return dist_df

    def pairwise_similarity_list(fp_list: List[DataStructs.ExplicitBitVect]) -> List[float]:
        """
        Calculate pairwise Tanimoto similarities for a list of RDKit fingerprint objects.

        Parameters:
            fp_list (List[DataStructs.ExplicitBitVect]): A list of RDKit fingerprint objects.

        Returns:
            List[float]: A flattened list of pairwise Tanimoto similarities.
        """
        if not fp_list or len(fp_list) < 2:
            raise ValueError("Input fingerprint list must contain at least two fingerprints.")

        similarity_list = []  # Create the empty list to store similarities
        for i in range(1, len(fp_list)):
            # Calculate Tanimoto similarities for the current fingerprint against previous ones
            tanimoto = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            similarity_list.append(tanimoto)

        # Flatten the list of lists into a 1D list
        all_distances = list(np.concatenate(similarity_list).flat)
        return all_distances


    def get_ecfp(smiles: str, radius: int = 2, nbits: int = 2048) -> DataStructs.ExplicitBitVect:
        """
        Generate an ECFP (Extended-Connectivity Fingerprint) for a given SMILES string.

        Parameters:
            smiles (str): A SMILES string to be fingerprinted.
            radius (int, optional): The Morgan fingerprint radius (default: 2).
            nbits (int, optional): The number of bits to use for hashing (default: 2048).

        Returns:
            DataStructs.ExplicitBitVect: A hashed ECFP RDKit fingerprint object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Generate the hashed Morgan fingerprint
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nbits)
        return fp



class ConfGen():
    '''A class for embedding and featurizing 3D conformers.

    Arguments
    ==========
    n_confs (int): Defaults to 50. The number of conformers to use for generation tasks.
    '''
    def __init__(self, n_confs:int = 50):
        self.n_confs = n_confs

        # specify the conformer generator using a specific seed
        self.conf_generator = AllChem.ETKDGv3()
        self.conf_generator = 0xf00d

        #Define featurefactory from defualt base features
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))

        #Create dict for holding the params for each feature
        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            self.fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = self.fparams

        #Define list the features to keep for pharmacophore mapping
        self.feats = ('Donor','Acceptor','NegIonizable','PosIonizable','Aromatic')

    def add_confs_to_df(self, df:pd.DataFrame, smiles_col:str, ):
        ''' Create a "Mols" column with embedded conformers from a smiles column and create a new column for 'mmff' forcefield data. 
        
        Arguments
        ==========
        df (pd.DataFrame): Required. A dataframe containing a smiles column.\\
        smiles_col (str): Required. The name of the column with the smiles to use.

        Returns
        ========
        df (pd.DataFrame): An updated dataframe containing the new columns "Mols" and "mmff"
        '''

        # Convert the smiles to mols and add them to the df
        df['Mols'] = [Chem.MolFromSmiles(x) for x in list(df[smiles_col])]

        # Add H's to the mols for 3D embedding
        df['Mols']=df['Mols'].apply(lambda mol:Chem.AddHs(mol))

        # Embed the mols using n conformations
        tqdm_notebook.pandas(desc="Embedding 3d Conformers")
        df['Mols'].progress_apply(lambda mol:AllChem.EmbedMultipleConfs(mol, self.n_confs, self.conf_generator))

        # Add mmff to the data_df
        tqdm_notebook.pandas(desc="Appliying Forcefields")
        df["mmff"]=df['Mols'].progress_apply(lambda mol:AllChem.MMFFGetMoleculeProperties(mol))

        return df
    
    def add_confs_to_mol(self, mol:Chem.Mol):
        '''Embed conformers to a mol and create the 'mmff' forcefield data. 
        
        Arguments
        ==========
        mol (rdkit mol): Required. A mol to generate conformers for.

        Returns
        ========
        mol (rdkit mol): An updated mol with embedded conformers.\\
        mmff (rdkit mmff): The forcefield properties for the conformers.
        '''

        mol=Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, self.n_confs, self.conf_generator)
        mmff = AllChem.MMFFGetMoleculeProperties(mol)
        return mol, mmff
    
    
    #Define a function to align a list of mols to a reference mol and visualize them
    def align_mols_to_ref(self, mols:list, ref_mol:Chem.Mol, mmff, ref_mmff):
            '''Perform and visualize 3D alignment of a the closest matching confor 
            
            Arguments
            ==========
            mols (list): Required. A list of mols with embedded 3D conformers.
            ref_mol (rdkit mol): Required. A mol with embedded conformers to use as an alignment template.
            mmff (list): Required. A list of mmff forcefield properties for the list of mols.
            ref_mmff (rdkit mmff): Required. The mmff forcefield properties for the ref_mol.

            Notes:\\
            The forcefield data allows a search for the closest matching (highest scoring) conformer.\\
            Using a ConfGen instance with a higher value of n_confs may improve the alignment results
            but will take more time to calculate.

            Returns
            ========
            3D view of the overlayed conformers
            '''
            # create a py3Dmol object to visualize the alignment
            O3A_alignment = py3Dmol.view(width=800, height=600)
            colors=('cyanCarbon',
                    'redCarbon',
                    'blueCarbon',
                    'aliceblueCarbon',
                    'greenCarbon',
                    'yellowCarbon',
                    'orangeCarbon',
                    'magentaCarbon',
                    'purpleCarbon',
                    'deeppinkCarbon')

            # Iterate over the mols and check the alignment score for each conformation
            for idx, mol in (enumerate(mols)):
                tempscore = []
                for conf_id in range(self.n_confs):
                    O3A = rdMolAlign.GetO3A(mol, ref_mol, mmff[idx], ref_mmff, conf_id, 0)
                    O3A.Align()
                    tempscore.append(O3A.Score())
                bestscore = np.argmax(tempscore)

                # Add the best scoring conformation to the viewer
                IPythonConsole.addMolToView(mol,O3A_alignment,confId=int(bestscore))
            # Iterate over the mols and set thier styles in the viewer
            for idx, mol in enumerate(mols):
                O3A_alignment.setStyle({'model':idx,}, {'stick':{'colorscheme':colors[idx%len(colors)]}})
            O3A_alignment.zoomTo()
            return O3A_alignment.show()

    def map_feats(self, df:pd.DataFrame, mol_col:str):
        '''Use the generators feature factory to map pharmacophore features to the mols in the specified dataframe.

        Arguments
        ==========
        df (pd.DataFrame): Required. A dataframe with mols containing embedded conformers.
        mol_col (str): Required. The name of the column with the mols.

        Returns
        ========
        featLists (list): A list containing a list of mapped pharmacophores for each mol.
        '''
        featLists = []
        for mol in df[mol_col]:
            rawFeats = self.fdef.GetFeaturesForMol(mol) #get the list of all features for the mol
            featLists.append([f for f in rawFeats if f.GetFamily() in self.feats]) # filter the full list to only include the ones in the keep list
        return featLists
    



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


def plot_smiles(df:pd.DataFrame, smi_col:str, ID_col:str, X_col:str, Y_col:str, img_col:str,  *args:str):
    output_notebook()
    plot_df=pd.DataFrame()
    plot_df['Smiles'] = df[smi_col]
    plot_df['ID'] = df[ID_col]
    plot_df[X_col] = df[X_col]
    plot_df[Y_col] = df[Y_col]
    plot_df['imgs'] = df[img_col]

    #add cols for the *args
    for col in args:
        if col in df.columns:
            plot_df[col] = df[col]
        else:
            print(f"Warning: The column '{col}' does not exist in the provided DataFrame.")
    
    #Define the bokeh data source
    source = ColumnDataSource(plot_df)

    #Define the tooltips layout
    TOOLTIPS = """\
    <div>
        <div>
            @imgs{safe}
        </div>
        <div>
            <span>@ID</span>
        </div>
    </div>"""

    #Create the plot
    p1 = figure(width=1000, height=1000, title=f'Smiles Plot of {X_col} vs {Y_col}')
    p1.circle(X_col, Y_col, source=source, size=10)
    p1.xaxis.axis_label = X_col
    p1.yaxis.axis_label = Y_col
    p1.add_tools(HoverTool(tooltips=TOOLTIPS))#Add tools to the plot
    

    # Output the plot to an HTML file
    output_file("SmilesPlot.html")
    show(p1)



def plot_smiles_color(
    df: pd.DataFrame, 
    smi_col: str, 
    ID_col: str, 
    X_col: str, 
    Y_col: str, 
    img_col: str, 
    color_col: str, 
    *args: str):

    """
    Plots a scatter plot of SMILES data with colored points based on a specified column using Bokeh.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecular data.
        smi_col (str): Column containing SMILES strings.
        ID_col (str): Column containing molecule IDs.
        X_col (str): Column for X-axis data.
        Y_col (str): Column for Y-axis data.
        img_col (str): Column containing image data for tooltips.
        color_col (str): Column for coloring the points.
        *args (str): Additional column names to include in the data source.

    Returns:
        None: Displays the plot in the notebook or saves it as an HTML file.
    """
    output_notebook()

    # Prepare the plotting DataFrame
    required_cols = [smi_col, ID_col, X_col, Y_col, img_col, color_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the DataFrame: {', '.join(missing_cols)}")

    plot_df = df[required_cols].copy()
    plot_df.columns = ['Smiles', 'ID', X_col, Y_col, 'imgs', 'Binding']

    # Remove rows with non-numeric values in the "Binding" column
    plot_df = plot_df[pd.to_numeric(plot_df['Binding'], errors='coerce').notna()]

    # Add additional columns if provided
    for col in args:
        if col in df.columns:
            plot_df[col] = df[col]
        else:
            print(f"Warning: The column '{col}' does not exist in the provided DataFrame.")
    
    # Define the Bokeh data source
    source = ColumnDataSource(plot_df)

    # Define the tooltips layout
    TOOLTIPS = """
    <div>
        <div>
            @imgs{safe}
        </div>
        <div>
            <span>@ID</span>
        </div>
    </div>
    """

    # Define the color mapper for the "Binding" column
    mapper = linear_cmap(
        field_name='Binding',
        palette=Turbo256,
        low=plot_df['Binding'].min(),
        high=plot_df['Binding'].max()
    )
    
    # Define the color bar legend
    color_bar = ColorBar(
        color_mapper=mapper['transform'],
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title='Binding'
    )

    # Create the plot
    p1 = figure(
        width=1000, 
        height=1000, 
        title=f"SMILES Plot of {X_col} vs {Y_col}"
    )
    p1.circle(
        X_col, 
        Y_col, 
        source=source, 
        size=6, 
        color=mapper
    )
    p1.xaxis.axis_label = X_col
    p1.yaxis.axis_label = Y_col
    p1.add_layout(color_bar, "right")
    p1.add_tools(HoverTool(tooltips=TOOLTIPS))
        
    # Show the plot
    output_file("SmilesPlot.html")
    show(p1)


def perform_umap(df: pd.DataFrame, arr_col: str, dims: int = 2, scaler_type: str = None):
    """
    Performs UMAP dimensionality reduction on array data from a specified DataFrame column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a column with array data.
        arr_col (str): Name of the column containing array data (e.g., fingerprints or descriptors).
        dims (int): Number of dimensions for UMAP reduction (default: 2).
        scaler_type (str): Type of scaling to apply before UMAP. Options are "standard", "minmax", or None (default: None).

    Returns:
        pd.DataFrame: Updated DataFrame with UMAP coordinate columns added.
        np.ndarray: UMAP coordinates as a NumPy array.
    """
    # Check if the column exists
    if arr_col not in df.columns:
        raise ValueError(f"Column '{arr_col}' not found in the DataFrame.")

    # Data Scaling
    data = list(df[arr_col])  # Default: no scaling
    if scaler_type:
        print(f"Scaling data with {scaler_type.capitalize()}Scaler...")
        if scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Use 'standard', 'minmax', or None.")
        data = scaler.fit_transform(data)

    # Perform UMAP
    print(f"Performing UMAP to reduce to {dims} dimensions...")
    reducer = umap.UMAP(n_components=dims)
    umap_result = reducer.fit_transform(data)

    # Add UMAP results to DataFrame
    for i in range(dims):
        df[f'umap_{i}'] = umap_result[:, i]

    return df, umap_result



def perform_pca(df: pd.DataFrame, arr_col: str, dims: int = 2, scaler_type: str = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Performs PCA dimensionality reduction on array data from a specified DataFrame column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a column with array data.
        arr_col (str): Name of the column containing array data (e.g., fingerprints or descriptors).
        dims (int): Number of dimensions for PCA reduction (default: 2).
        scaler_type (str): Type of scaling to apply before PCA. Options are "standard", "minmax", or None (default: None).

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 
            - Updated DataFrame with PCA coordinate columns added.
            - PCA coordinates as a NumPy array.
    """
    # Validate input parameters
    if arr_col not in df.columns:
        raise ValueError(f"Column '{arr_col}' not found in the DataFrame.")

    # Scaling options
    data = np.array(df[arr_col].tolist())  # Ensure input is in a suitable format
    if scaler_type:
        print(f"Scaling data with {scaler_type.capitalize()}Scaler...")
        if scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        elif scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Use 'standard', 'minmax', or None.")
        data = scaler.fit_transform(data)

    # Perform PCA
    print(f"Performing PCA to reduce to {dims} dimensions...")
    pca = PCA(n_components=dims)
    pca_result = pca.fit_transform(data)

    # Add PCA results to DataFrame
    for i in range(dims):
        df[f'pca_{i}'] = pca_result[:, i]

    return df, pca_result




def perform_kmeans(df: pd.DataFrame, ndarray: np.ndarray, n_clusters: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Performs K-Means clustering on a given ndarray of coordinates and adds cluster labels to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame to which cluster labels will be added.
        ndarray (np.ndarray): Array of coordinates to cluster (e.g., UMAP or PCA results).
        n_clusters (int): Number of clusters to form.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 
            - Updated DataFrame with a 'clusters' column containing cluster labels.
            - Cluster labels as a NumPy array.
    """
    # Validate inputs
    if not isinstance(ndarray, np.ndarray):
        raise ValueError("Input 'ndarray' must be a NumPy array.")
    if ndarray.shape[0] != len(df):
        raise ValueError("Number of rows in 'ndarray' must match the number of rows in the DataFrame.")
    if n_clusters <= 0:
        raise ValueError("Number of clusters 'n_clusters' must be greater than 0.")
    
    # Perform K-Means clustering
    print(f"Calculating K-Means clusters with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Adding random_state for reproducibility
    kmeans_labels = kmeans.fit_predict(ndarray)

    # Add cluster labels to the DataFrame
    df = df.copy()  # Avoids modifying the original DataFrame
    df['clusters'] = kmeans_labels

    return df, kmeans_labels


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




#Define functions for drawing the features
def colorToHex_old(rgb): #This was the original function that results in the same hex values for donors and acceptors
    rgb = [f'{int(255*x):x}' for x in rgb]
    return '0x'+''.join(rgb)

def colorToHex(rgb):
    rgb = [f'{int(255 * x):02x}' for x in rgb]
    return '0x' + ''.join(rgb)

def drawmolfeats(m, feats, p=None, confId=-1, removeHs=True):
        if p is None:
            p = py3Dmol.view(width=400, height=400)
        p.removeAllModels()
        if removeHs:
            m = Chem.RemoveHs(m)
        IPythonConsole.addMolToView(m,p,confId=confId)
        for feat in feats:
            pos = feat.GetPos()
            clr = featColors.get(feat.GetFamily(),(.5,.5,.5))
            p.addSphere({'center':{'x':pos.x,'y':pos.y,'z':pos.z},'radius':.5,'color':colorToHex(clr)});
        p.zoomTo()
        return p.show()

def drawfeatsonly(m, feats, p=None, confId=-1, removeHs=True):
        if p is None:
            p = py3Dmol.view(width=400, height=400)
        p.removeAllModels()
        if removeHs:
            m = Chem.RemoveHs(m)
        for feat in feats:
            pos = feat.GetPos()
            clr = featColors.get(feat.GetFamily(),(.5,.5,.5))
            p.addSphere({'center':{'x':pos.x,'y':pos.y,'z':pos.z},'radius':.5,'color':colorToHex(clr)});
        p.zoomTo()
        return p.show()
