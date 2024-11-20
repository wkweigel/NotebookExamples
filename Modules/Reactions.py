from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


class Reactant:
    ''' A class used to store smiles and reactive group information for a single reactant.'''
    def __init__(self, smiles:str, *args:str):
        #initialize all reactities to false
        self.reactivity_dict=   {
                                'alkyl_amine':False,
                                'aryl_amine':False,
                                'boc_amine':False,
                                'fmoc_amine':False,
                                'alkyl_alcohol':False,
                                'aryl_alcohol':False,
                                'alkyl_halide':False,
                                'aryl_halide':False,
                                'aldehyde':False,
                                'ketone':False,
                                'alkyne':False,
                                'carboxylic_acid':False,
                                'boronic_acid':False,
                                'boronic_ester':False,
                                'vinyl_alkene':False,
                                'azide':False,
                                'SAFE':False
                                }
        self.args=args
        self.smiles=smiles
        #Use the *args to specify the intended mode(s) of reactivity
        for arg in self.args:
            if arg in self.reactivity_dict.keys():
                self.reactivity_dict[arg]=True
            else:
                print(f'\'{arg}\' is not recognized as a valid reactive group.')
                print(f'Acceptable reactivity groups are: {list(self.reactivity_dict.keys())}')
        
        

class Reactions:

    #Define deprotection reaction SMARTS
    boc_deprotection_smarts='[N:1]C(=O)OC(C)(C)(C)>>[N:1]'
    Ar_boc_deprotection_smarts='[n:1]C(=O)OC(C)(C)(C)>>[nH:1]'
    ester_deprotection_smarts='[CX3:1](=[OX1:2])-[OX2:3]-[C](C)(C)(C)>>[C:1](=[O:2])-[O:3]'
    alcohol_deprotection_smarts='[*:1]-[OX2:3]-[C](C)(C)(C)>>[*:1]-[O:3]'

    #Create deprotection reactions from the SMARTS
    boc_deprotection_reaction=AllChem.ReactionFromSmarts(boc_deprotection_smarts)
    Ar_boc_deprotection_reaction=AllChem.ReactionFromSmarts(Ar_boc_deprotection_smarts)
    ester_deprotection_reaction=AllChem.ReactionFromSmarts(ester_deprotection_smarts)
    alcohol_deprotection_reaction=AllChem.ReactionFromSmarts(alcohol_deprotection_smarts)

    #Identify the reagent with the specified reactivity type.
    #The allows for the reagents to be provided to the reaction in any order. 
    def identify_reagent(self, reagent_types:list, *args:Reactant ):
        match=None
        for reactant in args:
            for type in reagent_types:
                if reactant.reactivity_dict[type]==True: #Check if the reactant has the needed reactivity type
                    match=reactant 
        if match==None:
            raise ValueError(f'No reagents found of type {reagent_types}')  
        else:
            return(match)

            #if any([reactant.reactivity_dict.get(reagent) for reagent in reagent_types])==True:
               # return(reactant)

    def amide_bond_formation(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        amine=self.identify_reagent(['alkyl_amine', 'aryl_amine'], reactant1, reactant2)
        carboxylic_acid=self.identify_reagent(['carboxylic_acid',], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        amine_mol=Chem.MolFromSmiles(amine.smiles)
        carboxylic_acid_mol=Chem.MolFromSmiles(carboxylic_acid.smiles)
        
        #block the existing amides from reacting as an amine
        amidep = Chem.MolFromSmarts('[N;$(NC=[O,S])]')
        for match in amine_mol.GetSubstructMatches(amidep):
            amine_mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
        
        #run the reaction
        product = reaction.RunReactants ([carboxylic_acid_mol, amine_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)

    def amine_displacement(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[N;!H0!$(NC=[!#6]):1].[#6:2][F,Cl,Br,I]>>[N:1][#6:2]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        amine=self.identify_reagent(['alkyl_amine', 'aryl_amine'], reactant1, reactant2)
        alkyl_halide=self.identify_reagent(['alkyl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        amine_mol=Chem.MolFromSmiles(amine.smiles)
        alkyl_halide_mol=Chem.MolFromSmiles(alkyl_halide.smiles)
        
        #block the existing amides from reacting as an amine
        amidep = Chem.MolFromSmarts('[N;$(NC=[O,S])]')
        for match in amine_mol.GetSubstructMatches(amidep):
            amine_mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
        
        #run the reaction
        product = reaction.RunReactants ([amine_mol, alkyl_halide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def alcohol_displacement(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[#6;$([#6]~[#6]);!$([#6]=O):2][#8;H1:3].[Cl,Br,I][#6;H2;$([#6]~[#6]):4]>>[CH2:4][O:3][#6:2]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        alcohol=self.identify_reagent(['alkyl_alcohol', 'aryl_alcohol'], reactant1, reactant2)
        alkyl_halide=self.identify_reagent(['alkyl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        alcohol_mol=Chem.MolFromSmiles(alcohol.smiles)
        alkyl_halide_mol=Chem.MolFromSmiles(alkyl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([alcohol_mol, alkyl_halide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def suzuki_coupling(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[#6;H0;D3;$([#6](~[#6])~[#6]):1]B(O)O.[#6;H0;D3;$([#6](~[#6])~[#6]):2][Cl,Br,I]>>[#6:2][#6:1]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        boronic_acid=self.identify_reagent(['boronic_acid'], reactant1, reactant2)
        aryl_halide=self.identify_reagent(['aryl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        boronic_acid_mol=Chem.MolFromSmiles(boronic_acid.smiles)
        aryl_halide_mol=Chem.MolFromSmiles(aryl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([boronic_acid_mol, aryl_halide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)

    def reductive_amination(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[#6:2](=[#8])(-[#6:1]).[#7;H2,H1:3]>>[#6:2](-[#6:1])-[#7:3]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        amine=self.identify_reagent(['alkyl_amine', 'aryl_amine'], reactant1, reactant2)
        aldehyde=self.identify_reagent(['aldehyde'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        amine_mol=Chem.MolFromSmiles(amine.smiles)
        aldehyde_mol=Chem.MolFromSmiles(aldehyde.smiles)

        #block the existing amides from reacting as an amine
        amidep = Chem.MolFromSmarts('[N;$(NC=[O,S])]')
        for match in amine_mol.GetSubstructMatches(amidep):
            amine_mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
        
        #run the reaction
        product = reaction.RunReactants ([aldehyde_mol, amine_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def heck_coupling(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction
        reaction_smarts = '[#6;c,$(C(=O)O),$(C#N):3][#6;H1:2]=[#6;H2:1].[#6;$([#6]=[#6]),$(c:c):4][Cl,Br,I]>>[#6:4]/[#6:1]=[#6:2]/[#6:3]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        alkene=self.identify_reagent(['vinyl_alkene'], reactant1, reactant2)
        aryl_halide=self.identify_reagent(['aryl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        alkene_mol=Chem.MolFromSmiles(alkene.smiles)
        aryl_halide_mol=Chem.MolFromSmiles(aryl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([alkene_mol, aryl_halide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def buchwald_hartwig_coupling(self, reactant1:Reactant, reactant2:Reactant):
        #Define the reaction 
        reaction_smarts = '[Cl,Br,I][c:1].[N;!$(N=*)&!$(N#*)&H1,H2:2]>>[c:1][N:2]'
        #reaction_smarts = '[Cl,Br,I][c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1].[N;$(NC)&!$(N=*)&!$([N-])&!$(N#*)&!$([ND3])&!$([ND4])&!$(N[c,O])&!$(N[C,S]=[S,O,N]),H2&$(Nc1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):2]>>[c:1][N:2]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        amine=self.identify_reagent(['alkyl_amine', 'aryl_amine'], reactant1, reactant2)
        aryl_halide=self.identify_reagent(['aryl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        amine_mol=Chem.MolFromSmiles(amine.smiles)
        aryl_halide_mol=Chem.MolFromSmiles(aryl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([aryl_halide_mol, amine_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def wittig_olefination(self, reactant1:Reactant, reactant2:Reactant):
        #Note: For simplicity, the halide synthon is used, not the phosphonium salt of the halide.
        #Define the reaction
        reaction_smarts = '[#6:3]-[C;H1,$([CH0](-[#6])[#6]);!$(CC=O):1]=[OD1].[Cl,Br,I][C;H2;$(C-[#6]);!$(CC[I,Br]);!$(CCO[CH3]):2]>>[C:3][C:1]=[C:2]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        ketone=self.identify_reagent(['aldehyde', 'ketone'], reactant1, reactant2)
        aryl_halide=self.identify_reagent(['alkyl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        ketone_mol=Chem.MolFromSmiles(ketone.smiles)
        aryl_halide_mol=Chem.MolFromSmiles(aryl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([ketone_mol, aryl_halide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def sonogashira_coupling(self, reactant1:Reactant, reactant2:Reactant):
        #Note: For simplicity, the halide synthon is used, not the phosphonium salt of the halide.
        #Define the reaction
        reaction_smarts = '[#6;$(C=C-[#6]),$(c:c):1][Br,I].[CH1;$(C#CC),$(C#Cc):2]>>[#6:1][C:2]'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        alkyne=self.identify_reagent(['alkyne'], reactant1, reactant2)
        aryl_halide=self.identify_reagent(['aryl_halide', 'alkyl_halide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        alkyne_mol=Chem.MolFromSmiles(alkyne.smiles)
        aryl_halide_mol=Chem.MolFromSmiles(aryl_halide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([aryl_halide_mol, alkyne_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    
    def click_reaction(self, reactant1:Reactant, reactant2:Reactant):
        #Note: For simplicity, the halide synthon is used, not the phosphonium salt of the halide.
        #Define the reaction
        reaction_smarts = '[C:1]#[$([CH1]):2].[$([#6]-N=[N+]=[-N]),$([#6]-[N-]-[N+]#N):3]-N~N~N>>[*:3]n1[c:2][c:1]nn1'
        reaction=AllChem.ReactionFromSmarts(reaction_smarts)

        #Identfy the reagents 
        alkyne=self.identify_reagent(['alkyne'], reactant1, reactant2)
        azide=self.identify_reagent(['azide'], reactant1, reactant2)
        
        #Create the mol objects from the reagents
        alkyne_mol=Chem.MolFromSmiles(alkyne.smiles)
        azide_mol=Chem.MolFromSmiles(azide.smiles)
        
        #run the reaction
        product = reaction.RunReactants ([alkyne_mol, azide_mol])

        #Convert the product to smiles
        product_smiles=Chem.MolToSmiles(product[0][0])
        return(product_smiles)
    


    
