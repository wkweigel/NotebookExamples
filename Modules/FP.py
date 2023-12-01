from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

def make_FP_csv(smiles, fp, output_path, data_name):
    mol_list=[Chem.MolFromSmiles(smile) for smile in smiles]
    if fp=='ECFP6':
        ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=1024) for x in mol_list]
        fp_list=[list(l) for l in ECFP6]
        output_df=pd.DataFrame()
        output_df['Smiles']=smiles
        output_df['FP']=fp_list
        output_df.to_csv(output_path+data_name+'_'+fp+'.csv')

def make_FP_dict(smiles_dict, fp, type):
    smiles=list(smiles_dict.values())
    names=list(smiles_dict.keys())
    types=[type]*len(names)
    mol_list=[Chem.MolFromSmiles(smile) for smile in smiles]

    if fp=='ECFP6':
        ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=1024) for x in mol_list]
        fp_list=[list(l) for l in ECFP6]
        fp_array = np.array(fp_list)
        #fp_name_dict=dict(zip(names,fp_array))
        fp_class_dict=dict(zip(fp_array,types))
        class_items=fp_class_dict.items()
        RF_list=list(fp_class_dict.items())
    
    return(fp_list,names,types,RF_list)


def calc_FP(Mols):
    ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=1024) for x in Mols]
    fp_list=[list(l) for l in ECFP6]
    fp_array = np.array(fp_list)
    fp_df= pd.DataFrame(fp_list)
    return(fp_df)

def prepare_FP_df(dict, cls=''):
    df=pd.DataFrame()
    df['ID']=list(dict.keys())
    df['Smiles']=list(dict.values())
    df['Class']=[cls]*len(df)
    df['ROMol']=df.Smiles.apply(Chem.MolFromSmiles)
    return(df)