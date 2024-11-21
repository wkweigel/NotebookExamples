# Standard library imports
import os
from copy import deepcopy

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import py3Dmol

# RDKit and cheminformatics
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem import rdMolAlign
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.Features.ShowFeats import _featColors as featColors
from rdkit.Chem.rdFMCS import FindMCS

# Progress bar
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

# Configure environment and settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IPythonConsole.ipython_3d = True
tqdm.pandas()


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

def display_multiple_pharm3D(df, mol_col:str, featLists:list, width=1000, height=2000, grid_size=(6, 3)):
    """
    Draws the superimposed 3D structures of a probe molecule and a reference molecule using py3Dmol.

    Parameters:
        df (DataFrame): DataFrame containing probe molecules. Assumes a column 'conformer_mol' with RDKit Mol objects.
        mol_col (str): The name of a mol column containing 3D conformers.
        featLists (list): A list the pharmacophore features for the mols in the dataframe.
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
            Mol = deepcopy(df.iloc[mol_position_on_df][mol_col])

            # Add the mol to the viewer
            view.addModel(Chem.MolToMolBlock(Mol), 'mol', viewer=(i, j))
            #IPythonConsole.addMolToView(m,p,confId=confId)
            for feat in featLists[mol_position_on_df]:
                pos = feat.GetPos()
                clr = featColors.get(feat.GetFamily(),(.5,.5,.5))
                view.addSphere({'center':{'x':pos.x,'y':pos.y,'z':pos.z},'radius':.5,'color':colorToHex(clr)});
            
            # Set styles for the viewer
            view.setStyle({'stick': {}}, viewer=(i, j))
            view.setStyle({'model': 0}, {'stick': {'colorscheme': 'greenCarbon'}}, viewer=(i, j))

    view.zoomTo()
    return view.render()
