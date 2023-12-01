from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import copy
import os

from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem import AllChem, DataStructs, RDConfig, PyMol, Draw, rdPartialCharges, rdRGroupDecomposition, rdqueries, rdDepictor
from rdkit.Chem.Draw import IPythonConsole, SimilarityMaps, rdMolDraw2D
IPythonConsole.molSize=(900,700)
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG,Image

from collections import defaultdict
from PIL import Image as pilImage
import io

def split_IDs_to_csv(data_list, ratio, MainDataSetName):
    '''
    
    '''
    trainIDs ,testIDs = train_test_split(data_list,test_size=ratio)
    test_output_df=pd.DataFrame()
    train_output_df=pd.DataFrame()
    train_output_df['TrainIDs']=trainIDs
    test_output_df['TestIDs']=testIDs
    train_output_df.to_csv(MainDataSetName+'_Train.csv')
    test_output_df.to_csv(MainDataSetName+'_Test.csv')



class MolData:
    r'''A class for extracting Smiles and Mol-IDs from a csv based on a Hit/Miss classification column.

    Parameters:
    ==========
    
    datafilepath: Path to a csv containing "Hit" or "Miss" classifications. Required col headers= "DEL_Smiles", "DEL_ID", "Class"
    '''
    def __init__(self, datafilepath):
        self.data_df=pd.read_csv(datafilepath)
        self.hit_smiles=[]
        self.miss_smiles=[]
        self.hit_IDs=[]
        self.miss_IDs=[]
        for i in range(len(self.data_df)):
            if(self.data_df["Class"].iloc[i]=='Hit'):
                self.hit_smiles.append(self.data_df["DEL_Smiles"].iloc[i])
                self.hit_IDs.append(self.data_df['DEL_ID'].iloc[i])
            elif(self.data_df["Class"].iloc[i]=="Miss"):
                self.miss_smiles.append(self.data_df["DEL_Smiles"].iloc[i])
                self.miss_IDs.append(self.data_df['DEL_ID'].iloc[i])



class Randomized_ML_Prep:
    def __init__(self, datafilepath):
        self.data_df=pd.read_csv(datafilepath)
        self.all_smiles=self.data_df["DEL_Smiles"]
        self.all_IDs=self.data_df['DEL_ID']
        self.hit_smiles=[]
        self.miss_smiles=[]
        self.hit_IDs=[]
        self.miss_IDs=[]
        
        for i in range(len(self.data_df)):
            if(self.data_df["total score"].iloc[i]==12):
                self.hit_smiles.append(self.data_df["DEL_Smiles"].iloc[i])
                self.hit_IDs.append(self.data_df['DEL_ID'].iloc[i])
            elif(self.data_df["total score"].iloc[i]==0):
                self.miss_smiles.append(self.data_df["DEL_Smiles"].iloc[i])
                self.miss_IDs.append(self.data_df['DEL_ID'].iloc[i])
        
        self.compound_dict=dict(zip(self.all_IDs,self.all_smiles))

class Specific_ML_Prep:
    def __init__(self, datafilepath,test_hitIDpath,test_missIDpath,train_hitIDpath,train_missIDpath):
        
        #Import the CSVs
        self.data_df=pd.read_csv(datafilepath)
        self.test_hit_df=pd.read_csv(test_hitIDpath)
        self.test_miss_df=pd.read_csv(test_missIDpath)
        self.train_hit_df=pd.read_csv(train_hitIDpath)
        self.train_miss_df=pd.read_csv(train_missIDpath)

        #Create dict for holding IDs:Smiles
        self.compound_dict=dict(zip(self.data_df['DEL_ID'],self.data_df["DEL_Smiles"]))
        self.hit_testing_dict=dict(zip(self.test_hit_df['TestIDs'], [self.compound_dict[x] for x in self.test_hit_df['TestIDs']]))
        self.hit_training_dict=dict(zip(self.train_hit_df['TrainIDs'], [self.compound_dict[x] for x in self.train_hit_df['TrainIDs']]))
        self.miss_testing_dict=dict(zip(self.test_miss_df['TestIDs'], [self.compound_dict[x] for x in self.test_miss_df['TestIDs']]))
        self.miss_training_dict=dict(zip(self.train_miss_df['TrainIDs'], [self.compound_dict[x] for x in self.train_miss_df['TrainIDs']]))

        self.test_hit_smiles=[self.compound_dict[x] for x in self.test_hit_df['TestIDs']]
        self.test_miss_smiles=[self.compound_dict[x] for x in self.test_miss_df['TestIDs']]
        self.train_hit_smiles=[self.compound_dict[x] for x in self.train_hit_df['TrainIDs']]
        self.train_miss_smiles=[self.compound_dict[x] for x in self.train_miss_df['TrainIDs']]


    #Define helper functions
    def aligned_imgs_from_dict(self, input_dict,type='', output_location=''):
        names=list(input_dict.keys())
        self.prepare_alignment_df(input_dict)
        self.Output_Aligned_Core(self.mms,self.groups,self.qcore,('R1','R2'),filepath=output_location,legends=None,subImageSize=(300,300),filenames=names)

    def aligned_highlighted_imgs_from_dict(self, input_dict,output_location=''):
        names=list(input_dict.keys())
        self.prepare_alignment_df(input_dict)
        self.Output_Highlighted_BBs(self.mms,self.groups,self.qcore,('R1','R2'),filepath=output_location,legends=None,subImageSize=(300,300),filenames=names)

    def aligned_charge_imgs_from_dict(self, input_dict,output_location=''):
        names=list(input_dict.keys())
        self.prepare_alignment_df(input_dict)
        self.Output_Chargemaps(self.mms,self.groups,self.qcore,('R1','R2'),filepath=output_location,legends=None,subImageSize=(300,300),filenames=names)

    def unaligned_imgs_from_dict(self,dict,output_location=''):
            for id, smiles in dict.items():
                mol=Chem.MolFromSmiles(smiles)
                mol_ID=id
                Chem.Draw.MolToFile(mol, output_location + str(mol_ID) + '.png', size=(300,300))


    def make_mol_imgs(self, type, img_output_path):
        if type=='train':
            for id, smiles in self.hit_training_dict.items():
                mol=Chem.MolFromSmiles(smiles)
                mol_ID=id
                Chem.Draw.MolToFile(mol, img_output_path +'/hits/'+ str(mol_ID) + '.png',size=(300,300))

            for id, smiles in self.miss_training_dict.items():
                mol=Chem.MolFromSmiles(smiles)
                mol_ID=id
                Chem.Draw.MolToFile(mol, img_output_path +'/misses/' + str(mol_ID) + '.png',size=(300,300))

        if type=='test':
            for id, smiles in self.hit_testing_dict.items():
                mol=Chem.MolFromSmiles(smiles)
                mol_ID=id
                Chem.Draw.MolToFile(mol, img_output_path +'/hits/'+  str(mol_ID) + '.png',size=(300,300))

            for id, smiles in self.miss_testing_dict.items():
                mol=Chem.MolFromSmiles(smiles)
                mol_ID=id
                Chem.Draw.MolToFile(mol, img_output_path +'/misses/' +  str(mol_ID) + '.png',size=(300,300))


    def prepare_alignment_df(self, inputdict):
        rdDepictor.SetPreferCoordGen(True)
        temp_df=pd.DataFrame()
        temp_df['IDs']=list(inputdict.keys())
        temp_df['Smiles']=list(inputdict.values())
        temp_df['ROMol'] = temp_df.Smiles.apply(Chem.MolFromSmiles) #Create mols
        self.mols= temp_df['ROMol']
        for mol in self.mols:
            rdDepictor.Compute2DCoords(mol)

        self.core = Chem.MolFromSmiles('CNC(=O)[C@H](CNC([*:1])=O)NC([*:2])=O')
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(self.core)

        ps = Chem.AdjustQueryParameters.NoAdjustments()
        ps.makeDummiesQueries=True
        self.qcore = Chem.AdjustQueryProperties(self.core,ps)
        self.mhs = [Chem.AddHs(x,addCoords=True) for x in self.mols]
        self.mms = [x for x in self.mhs if x.HasSubstructMatch(self.qcore)]
        for m in self.mms:
            for atom in m.GetAtoms():
                atom.SetIntProp("SourceAtomIdx",atom.GetIdx())
        print(len(self.mhs),len(self.mms))
        RDLogger.DisableLog('rdApp.warning')
        self.groups,_ = rdRGroupDecomposition.RGroupDecompose([self.qcore],self.mms,asSmiles=False,asRows=True)
        #return(self.mols,self.mms,self.qcore)

    filepath='AlignedCpds/'

    def Highlighted_BBs(self, mol,row,core,width=300,height=300,
                        fillRings=True,legend="",
                        sourceIdxProperty="SourceAtomIdx",
                        lbls=('R1','R2'),filename="Molecule.png"):
        # copy the molecule and core
        mol = Chem.Mol(mol)
        core = Chem.Mol(core)

        

        # -------------------------------------------
        # include the atom map numbers in the substructure search in order to 
        # try to ensure a good alignment of the molecule to symmetric cores
        for at in core.GetAtoms():
            if at.GetAtomMapNum():
                at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200+at.GetAtomMapNum()))
                
        for lbl in row:
            if lbl=='Core':
                continue
            rg = row[lbl]
            for at in rg.GetAtoms():
                if not at.GetAtomicNum() and at.GetAtomMapNum() and \
                at.HasProp('dummyLabel') and at.GetProp('dummyLabel')==lbl:
                    # attachment point. the atoms connected to this
                    # should be from the molecule
                    for nbr in at.GetNeighbors():
                        if nbr.HasProp(sourceIdxProperty):
                            mAt = mol.GetAtomWithIdx(nbr.GetIntProp(sourceIdxProperty))
                            if mAt.GetIsotope():
                                mAt.SetIntProp('_OrigIsotope',mAt.GetIsotope())
                            mAt.SetIsotope(200+at.GetAtomMapNum())
        # remove unmapped hs so that they don't mess up the depiction
        rhps = Chem.RemoveHsParameters()
        rhps.removeMapped = False
        tmol = Chem.RemoveHs(mol,rhps)
        rdDepictor.GenerateDepictionMatching2DStructure(tmol,core)

        oldNewAtomMap={}
        # reset the original isotope values and account for the fact that
        # removing the Hs changed atom indices
        for i,at in enumerate(tmol.GetAtoms()):
            if at.HasProp(sourceIdxProperty):
                oldNewAtomMap[at.GetIntProp(sourceIdxProperty)] = i
                if at.HasProp("_OrigIsotope"):
                    at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                    at.ClearProp("_OrigIsotope")
                else:
                    at.SetIsotope(0)
        
        # ------------------
        #  set up our colormap
        #   the three choices here are all "colorblind" colormaps
        
        # "Tol" colormap from https://davidmathlogic.com/colorblind
        colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
        # "IBM" colormap from https://davidmathlogic.com/colorblind
        colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
        # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
        colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]
        for i,x in enumerate(colors):
            colors[i] = tuple(y/255 for y in x)

        #----------------------
        # Identify and store which atoms, bonds, and rings we'll be highlighting
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        atomrads = {}
        widthmults = {}

        rings = []
        for i,lbl in enumerate(lbls):    
            color = colors[i%len(colors)]
            rquery = row[lbl]
            Chem.GetSSSR(rquery)
            rinfo = rquery.GetRingInfo()
            for at in rquery.GetAtoms():
                if at.HasProp(sourceIdxProperty):
                    origIdx = oldNewAtomMap[at.GetIntProp(sourceIdxProperty)]
                    highlightatoms[origIdx].append(color)
                    atomrads[origIdx] = 0.4
            if fillRings:
                for aring in rinfo.AtomRings():
                    tring = []
                    allFound = True
                    for aid in aring:
                        at = rquery.GetAtomWithIdx(aid)
                        if not at.HasProp(sourceIdxProperty):
                            allFound = False
                            break
                        tring.append(oldNewAtomMap[at.GetIntProp(sourceIdxProperty)])
                    if allFound:
                        rings.append((tring,color))
            for qbnd in rquery.GetBonds():
                batom = qbnd.GetBeginAtom()
                eatom = qbnd.GetEndAtom()
                if batom.HasProp(sourceIdxProperty) and eatom.HasProp(sourceIdxProperty):
                    origBnd = tmol.GetBondBetweenAtoms(oldNewAtomMap[batom.GetIntProp(sourceIdxProperty)],
                                                    oldNewAtomMap[eatom.GetIntProp(sourceIdxProperty)])
                    bndIdx = origBnd.GetIdx()
                    highlightbonds[bndIdx].append(color)
                    widthmults[bndIdx] = 2

        d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
        dos = d2d.drawOptions()
        dos.useBWAtomPalette()
                    
        #----------------------
        # if we are filling rings, go ahead and do that first so that we draw
        # the molecule on top of the filled rings
        if fillRings and rings:
            # a hack to set the molecule scale
            d2d.DrawMoleculeWithHighlights(tmol,legend,dict(highlightatoms),
                                        dict(highlightbonds),
                                        atomrads,widthmults)
            d2d.ClearDrawing()
            conf = tmol.GetConformer()
            for (aring,color) in rings:
                ps = []
                for aidx in aring:
                    pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                    ps.append(pos)
                d2d.SetFillPolys(True)
                d2d.SetColour(color)
                d2d.DrawPolygon(ps)
            dos.clearBackground = False

        #----------------------
        # now draw the molecule, with highlights:
        d2d.DrawMoleculeWithHighlights(tmol,legend,dict(highlightatoms),dict(highlightbonds),
                                    atomrads,widthmults)
        d2d.FinishDrawing()
        d2d.WriteDrawingText(filename)
        

    #@interact(idx=range(0,len(self.mols)))
    #def draw_it(idx=0):
    #    m = self.mms[idx]
    #   row = groups[idx]
    #   return Image(highlight_rgroups(m,row,qcore,lbls=('R1','R2')))
    

    def Output_Highlighted_BBs(self,ms,groups,qcore,lbls,legends=None,nPerRow=4,subImageSize=(300,300),filepath='AlignedCpds/',filenames=''):

        imgSize = (subImageSize[0],subImageSize[1])
        res = pilImage.new('RGB',imgSize)
        
        for i,m in enumerate(ms):

            if legends:
                legend = legends[i]
                name=filenames[i]
            else:
                legend = ''
                name=filenames[i]
            png = self.Highlighted_BBs(m,groups[i],
                                                qcore,
                                                lbls=lbls,
                                                legend=legend,
                                                width=subImageSize[0],
                                                height=subImageSize[1],
                                                filename=filepath + name +'.png')

    def Align_Core(self, mol,row,core,width=300,height=300,
                        fillRings=True,legend="",
                        sourceIdxProperty="SourceAtomIdx",
                        lbls=('R1','R2'),filename="Molecule.png"):
        # copy the molecule and core
        mol = Chem.Mol(mol)
        core = Chem.Mol(core)

        

        # -------------------------------------------
        # include the atom map numbers in the substructure search in order to 
        # try to ensure a good alignment of the molecule to symmetric cores
        for at in core.GetAtoms():
            if at.GetAtomMapNum():
                at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200+at.GetAtomMapNum()))
                
        for lbl in row:
            if lbl=='Core':
                continue
            rg = row[lbl]
            for at in rg.GetAtoms():
                if not at.GetAtomicNum() and at.GetAtomMapNum() and \
                at.HasProp('dummyLabel') and at.GetProp('dummyLabel')==lbl:
                    # attachment point. the atoms connected to this
                    # should be from the molecule
                    for nbr in at.GetNeighbors():
                        if nbr.HasProp(sourceIdxProperty):
                            mAt = mol.GetAtomWithIdx(nbr.GetIntProp(sourceIdxProperty))
                            if mAt.GetIsotope():
                                mAt.SetIntProp('_OrigIsotope',mAt.GetIsotope())
                            mAt.SetIsotope(200+at.GetAtomMapNum())
        # remove unmapped hs so that they don't mess up the depiction
        rhps = Chem.RemoveHsParameters()
        rhps.removeMapped = False
        tmol = Chem.RemoveHs(mol,rhps)
        rdDepictor.GenerateDepictionMatching2DStructure(tmol,core)

        oldNewAtomMap={}
        # reset the original isotope values and account for the fact that
        # removing the Hs changed atom indices
        for i,at in enumerate(tmol.GetAtoms()):
            if at.HasProp(sourceIdxProperty):
                oldNewAtomMap[at.GetIntProp(sourceIdxProperty)] = i
                if at.HasProp("_OrigIsotope"):
                    at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                    at.ClearProp("_OrigIsotope")
                else:
                    at.SetIsotope(0)
        
        # ------------------
        #  set up our colormap
        #   the three choices here are all "colorblind" colormaps
        
        # "Tol" colormap from https://davidmathlogic.com/colorblind
        colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
        # "IBM" colormap from https://davidmathlogic.com/colorblind
        colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
        # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
        colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]
        for i,x in enumerate(colors):
            colors[i] = tuple(y/255 for y in x)

        #----------------------
        # Identify and store which atoms, bonds, and rings we'll be highlighting
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        atomrads = {}
        widthmults = {}

        rings = []
        for i,lbl in enumerate(lbls):    
            color = colors[i%len(colors)]
            rquery = row[lbl]
            Chem.GetSSSR(rquery)
            rinfo = rquery.GetRingInfo()
            for at in rquery.GetAtoms():
                if at.HasProp(sourceIdxProperty):
                    origIdx = oldNewAtomMap[at.GetIntProp(sourceIdxProperty)]
                    highlightatoms[origIdx].append(color)
                    atomrads[origIdx] = 0.4
            if fillRings:
                for aring in rinfo.AtomRings():
                    tring = []
                    allFound = True
                    for aid in aring:
                        at = rquery.GetAtomWithIdx(aid)
                        if not at.HasProp(sourceIdxProperty):
                            allFound = False
                            break
                        tring.append(oldNewAtomMap[at.GetIntProp(sourceIdxProperty)])
                    if allFound:
                        rings.append((tring,color))
            for qbnd in rquery.GetBonds():
                batom = qbnd.GetBeginAtom()
                eatom = qbnd.GetEndAtom()
                if batom.HasProp(sourceIdxProperty) and eatom.HasProp(sourceIdxProperty):
                    origBnd = tmol.GetBondBetweenAtoms(oldNewAtomMap[batom.GetIntProp(sourceIdxProperty)],
                                                    oldNewAtomMap[eatom.GetIntProp(sourceIdxProperty)])
                    bndIdx = origBnd.GetIdx()
                    highlightbonds[bndIdx].append(color)
                    widthmults[bndIdx] = 2

        d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
        dos = d2d.drawOptions()
        dos.useBWAtomPalette()
                    
 

        #----------------------
        # now draw the molecule, with highlights:
        d2d.DrawMolecule(tmol)
        d2d.FinishDrawing()
        d2d.WriteDrawingText(filename)
        

    #@interact(idx=range(0,len(self.mols)))
    #def draw_it(idx=0):
    #    m = self.mms[idx]
    #   row = groups[idx]
    #   return Image(highlight_rgroups(m,row,qcore,lbls=('R1','R2')))
    

    def Output_Aligned_Core(self,ms,groups,qcore,lbls,legends=None,nPerRow=4,subImageSize=(300,300),filepath='AlignedCpds/',filenames=''):

        imgSize = (subImageSize[0],subImageSize[1])
        res = pilImage.new('RGB',imgSize)
        
        for i,m in enumerate(ms):

            if legends:
                legend = legends[i]
                name=filenames[i]
            else:
                legend = ''
                name=filenames[i]
            png = self.Align_Core(m,groups[i],
                                                qcore,
                                                lbls=lbls,
                                                legend=legend,
                                                width=subImageSize[0],
                                                height=subImageSize[1],
                                                filename=filepath + name +'.png')

    def show_png(self, data):
            from io import BytesIO
            bio = io.BytesIO(data)
            img = pilImage.open(bio)
            return img

    def Make_Chargemap(self, mol,row,core,width=300,height=300,
                        fillRings=True,legend="",
                        sourceIdxProperty="SourceAtomIdx",
                        lbls=('R1','R2'),filename="Molecule.png"):
        # copy the molecule and core
        mol = Chem.Mol(mol)
        core = Chem.Mol(core)

        

        # -------------------------------------------
        # include the atom map numbers in the substructure search in order to 
        # try to ensure a good alignment of the molecule to symmetric cores
        for at in core.GetAtoms():
            if at.GetAtomMapNum():
                at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200+at.GetAtomMapNum()))
                
        for lbl in row:
            if lbl=='Core':
                continue
            rg = row[lbl]
            for at in rg.GetAtoms():
                if not at.GetAtomicNum() and at.GetAtomMapNum() and \
                at.HasProp('dummyLabel') and at.GetProp('dummyLabel')==lbl:
                    # attachment point. the atoms connected to this
                    # should be from the molecule
                    for nbr in at.GetNeighbors():
                        if nbr.HasProp(sourceIdxProperty):
                            mAt = mol.GetAtomWithIdx(nbr.GetIntProp(sourceIdxProperty))
                            if mAt.GetIsotope():
                                mAt.SetIntProp('_OrigIsotope',mAt.GetIsotope())
                            mAt.SetIsotope(200+at.GetAtomMapNum())
        # remove unmapped hs so that they don't mess up the depiction
        rhps = Chem.RemoveHsParameters()
        rhps.removeMapped = False
        tmol = Chem.RemoveHs(mol,rhps)
        rdDepictor.GenerateDepictionMatching2DStructure(tmol,core)

        oldNewAtomMap={}
        # reset the original isotope values and account for the fact that
        # removing the Hs changed atom indices
        for i,at in enumerate(tmol.GetAtoms()):
            if at.HasProp(sourceIdxProperty):
                oldNewAtomMap[at.GetIntProp(sourceIdxProperty)] = i
                if at.HasProp("_OrigIsotope"):
                    at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                    at.ClearProp("_OrigIsotope")
                else:
                    at.SetIsotope(0)
        
        # ------------------
        #  set up our colormap
        #   the three choices here are all "colorblind" colormaps
        
        # "Tol" colormap from https://davidmathlogic.com/colorblind
        colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
        # "IBM" colormap from https://davidmathlogic.com/colorblind
        colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
        # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
        colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]
        for i,x in enumerate(colors):
            colors[i] = tuple(y/255 for y in x)

        #----------------------
        # Identify and store which atoms, bonds, and rings we'll be highlighting
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        atomrads = {}
        widthmults = {}

        rings = []
        for i,lbl in enumerate(lbls):    
            color = colors[i%len(colors)]
            rquery = row[lbl]
            Chem.GetSSSR(rquery)
            rinfo = rquery.GetRingInfo()
            for at in rquery.GetAtoms():
                if at.HasProp(sourceIdxProperty):
                    origIdx = oldNewAtomMap[at.GetIntProp(sourceIdxProperty)]
                    highlightatoms[origIdx].append(color)
                    atomrads[origIdx] = 0.4
            if fillRings:
                for aring in rinfo.AtomRings():
                    tring = []
                    allFound = True
                    for aid in aring:
                        at = rquery.GetAtomWithIdx(aid)
                        if not at.HasProp(sourceIdxProperty):
                            allFound = False
                            break
                        tring.append(oldNewAtomMap[at.GetIntProp(sourceIdxProperty)])
                    if allFound:
                        rings.append((tring,color))
            for qbnd in rquery.GetBonds():
                batom = qbnd.GetBeginAtom()
                eatom = qbnd.GetEndAtom()
                if batom.HasProp(sourceIdxProperty) and eatom.HasProp(sourceIdxProperty):
                    origBnd = tmol.GetBondBetweenAtoms(oldNewAtomMap[batom.GetIntProp(sourceIdxProperty)],
                                                    oldNewAtomMap[eatom.GetIntProp(sourceIdxProperty)])
                    bndIdx = origBnd.GetIdx()
                    highlightbonds[bndIdx].append(color)
                    widthmults[bndIdx] = 2

        d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
        dos = d2d.drawOptions()
        dos.useBWAtomPalette()
                    
        #Calculate Charges
        rdPartialCharges.ComputeGasteigerCharges(tmol)
        chgs = [x.GetDoubleProp("_GasteigerCharge") for x in tmol.GetAtoms()]

        #----------------------
        # now draw the molecule, with charges:
        SimilarityMaps.GetSimilarityMapFromWeights(tmol,chgs,draw2d=d2d)
        d2d.FinishDrawing()
        self.show_png(d2d.GetDrawingText())
        d2d.WriteDrawingText(filename)

        

    #@interact(idx=range(0,len(self.mols)))
    #def draw_it(idx=0):
    #    m = self.mms[idx]
    #   row = groups[idx]
    #   return Image(highlight_rgroups(m,row,qcore,lbls=('R1','R2')))
    

    def Output_Chargemaps(self,ms,groups,qcore,lbls,legends=None,nPerRow=4,subImageSize=(300,300),filepath='AlignedCpds/',filenames=''):

        imgSize = (subImageSize[0],subImageSize[1])
        res = pilImage.new('RGB',imgSize)
        
        for i,m in enumerate(ms):

            if legends:
                legend = legends[i]
                name=filenames[i]
            else:
                legend = ''
                name=filenames[i]
            png = self.Make_Chargemap(m,groups[i],
                                                qcore,
                                                lbls=lbls,
                                                legend=legend,
                                                width=subImageSize[0],
                                                height=subImageSize[1],
                                                filename=filepath + name +'.png')
    

    def Chargemap_w_Highlights(self, mol,row,core,width=300,height=300,
                        fillRings=True,legend="",
                        sourceIdxProperty="SourceAtomIdx",
                        lbls=('R1','R2'),filename="Molecule.png"):
        # copy the molecule and core
        mol = Chem.Mol(mol)
        core = Chem.Mol(core)

        

        # -------------------------------------------
        # include the atom map numbers in the substructure search in order to 
        # try to ensure a good alignment of the molecule to symmetric cores
        for at in core.GetAtoms():
            if at.GetAtomMapNum():
                at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200+at.GetAtomMapNum()))
                
        for lbl in row:
            if lbl=='Core':
                continue
            rg = row[lbl]
            for at in rg.GetAtoms():
                if not at.GetAtomicNum() and at.GetAtomMapNum() and \
                at.HasProp('dummyLabel') and at.GetProp('dummyLabel')==lbl:
                    # attachment point. the atoms connected to this
                    # should be from the molecule
                    for nbr in at.GetNeighbors():
                        if nbr.HasProp(sourceIdxProperty):
                            mAt = mol.GetAtomWithIdx(nbr.GetIntProp(sourceIdxProperty))
                            if mAt.GetIsotope():
                                mAt.SetIntProp('_OrigIsotope',mAt.GetIsotope())
                            mAt.SetIsotope(200+at.GetAtomMapNum())
        # remove unmapped hs so that they don't mess up the depiction
        rhps = Chem.RemoveHsParameters()
        rhps.removeMapped = False
        tmol = Chem.RemoveHs(mol,rhps)
        rdDepictor.GenerateDepictionMatching2DStructure(tmol,core)

        oldNewAtomMap={}
        # reset the original isotope values and account for the fact that
        # removing the Hs changed atom indices
        for i,at in enumerate(tmol.GetAtoms()):
            if at.HasProp(sourceIdxProperty):
                oldNewAtomMap[at.GetIntProp(sourceIdxProperty)] = i
                if at.HasProp("_OrigIsotope"):
                    at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                    at.ClearProp("_OrigIsotope")
                else:
                    at.SetIsotope(0)
        
        # ------------------
        #  set up our colormap
        #   the three choices here are all "colorblind" colormaps
        
        # "Tol" colormap from https://davidmathlogic.com/colorblind
        colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
        # "IBM" colormap from https://davidmathlogic.com/colorblind
        colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
        # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
        colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]
        for i,x in enumerate(colors):
            colors[i] = tuple(y/255 for y in x)

        #----------------------
        # Identify and store which atoms, bonds, and rings we'll be highlighting
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        atomrads = {}
        widthmults = {}

        rings = []
        for i,lbl in enumerate(lbls):    
            color = colors[i%len(colors)]
            rquery = row[lbl]
            Chem.GetSSSR(rquery)
            rinfo = rquery.GetRingInfo()
            for at in rquery.GetAtoms():
                if at.HasProp(sourceIdxProperty):
                    origIdx = oldNewAtomMap[at.GetIntProp(sourceIdxProperty)]
                    highlightatoms[origIdx].append(color)
                    atomrads[origIdx] = 0.4
            if fillRings:
                for aring in rinfo.AtomRings():
                    tring = []
                    allFound = True
                    for aid in aring:
                        at = rquery.GetAtomWithIdx(aid)
                        if not at.HasProp(sourceIdxProperty):
                            allFound = False
                            break
                        tring.append(oldNewAtomMap[at.GetIntProp(sourceIdxProperty)])
                    if allFound:
                        rings.append((tring,color))
            for qbnd in rquery.GetBonds():
                batom = qbnd.GetBeginAtom()
                eatom = qbnd.GetEndAtom()
                if batom.HasProp(sourceIdxProperty) and eatom.HasProp(sourceIdxProperty):
                    origBnd = tmol.GetBondBetweenAtoms(oldNewAtomMap[batom.GetIntProp(sourceIdxProperty)],
                                                    oldNewAtomMap[eatom.GetIntProp(sourceIdxProperty)])
                    bndIdx = origBnd.GetIdx()
                    highlightbonds[bndIdx].append(color)
                    widthmults[bndIdx] = 2

            d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
            dos = d2d.drawOptions()
            dos.useBWAtomPalette()
                        
            #Calculate Charges
            rdPartialCharges.ComputeGasteigerCharges(tmol)
            chgs = [x.GetDoubleProp("_GasteigerCharge") for x in tmol.GetAtoms()]

            #----------------------
            # now draw the molecule, with charges:
            SimilarityMaps.GetSimilarityMapFromWeights(tmol,chgs,draw2d=d2d)
            d2d.FinishDrawing()
            self.show_png(d2d.GetDrawingText())
            d2d.WriteDrawingText(filename)

            #----------------------
            # now draw the molecule, with highlights:
            d2d.DrawMoleculeWithHighlights(tmol,legend,dict(highlightatoms),dict(highlightbonds),
                                        atomrads,widthmults)
            d2d.FinishDrawing()
            d2d.WriteDrawingText(filename)
        

            #@interact(idx=range(0,len(self.mols)))
            #def draw_it(idx=0):
            #    m = self.mms[idx]
            #   row = groups[idx]
            #   return Image(highlight_rgroups(m,row,qcore,lbls=('R1','R2')))
        

        def Output_Chargemap_w_Highlights(self,ms,groups,qcore,lbls,legends=None,nPerRow=4,subImageSize=(300,300),filepath='AlignedCpds/',filenames=''):

            imgSize = (subImageSize[0],subImageSize[1])
            res = pilImage.new('RGB',imgSize)
            
            for i,m in enumerate(ms):

                if legends:
                    legend = legends[i]
                    name=filenames[i]
                else:
                    legend = ''
                    name=filenames[i]
                png = self.charge_highlighted_rgroups_output(m,groups[i],
                                                    qcore,
                                                    lbls=lbls,
                                                    legend=legend,
                                                    width=subImageSize[0],
                                                    height=subImageSize[1],
                                                    filename=filepath + name +'.png')