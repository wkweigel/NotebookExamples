# Standard library imports
import os
from typing import Tuple, Optional, List

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Viridis256, Turbo256, Inferno256

# Machine learning and clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap

# RDKit and cheminformatics
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors,rdFMCS,rdMolDescriptors,rdMolAlign
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdFMCS import FindMCS

# MHFP and molecular fingerprints
from mhfp.encoder import MHFPEncoder
from mxfp import mxfp

# Progress bar
from tqdm import tqdm

# Configure environment and settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IPythonConsole.ipython_3d = True
tqdm.pandas()



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