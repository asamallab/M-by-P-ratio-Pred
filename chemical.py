#Calculation Tanimoto coeff. and generation of 2D structures
#Following analysis requires RDKit
#sudo apt-get install python-rdkit librdkit1 rdkit-data
#R.P. Vivek-Ananth, IMSc
#Last modified: 28-Apr-2017
#Last modified: Wed Nov  4 16:20:51 IST 2020

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys

#output file with similarity. col1: molecule1; col2: molecule2; col3: Tanimoto ECFP4; col4: Tanimoto ECFC4; col5: Tanimoto RDKit (Daylight); col6: Tanimoto Maccs Key.

#reading molecules from sdf file
edcs1 = Chem.SDMolSupplier('combined_1335.sdf')
#edcs1 = Chem.SDMolSupplier('14.sdf')
edcs2 = Chem.SDMolSupplier('combined_1335.sdf')

print ('Number of molecules {}'.format(len(edcs1)))
print ('Number of molecules {}'.format(len(edcs2)))

#calculation of fingerprints
def finger(edcs,source):
    figprnts_ecfc4={}
    figprnts_maccs={}
    molnames=[]
    for mol in edcs:
        if mol is None:
            print('!!! Error: {}'.format(t1))
        else:
            #print(mol.GetProp('_Name'),mol.GetNumAtoms())
            if source == 'user':
                t1=mol.GetProp('_Name')
            elif source == 'drugbank':
                t1=mol.GetProp('DATABASE_ID')
            molnames.append(t1)
            figprnts_ecfc4[t1]=AllChem.GetMorganFingerprint(mol,2)#ECFC4
            figprnts_maccs[t1]=MACCSkeys.GenMACCSKeys(mol)#MACCS
    return figprnts_ecfc4,figprnts_maccs,molnames


#calculation of tanimoto coefficient
def tc(fe1,fm1,fe2,fm2, mol1, mol2,out):
    fo=open(out,'w')
    for i in mol1:
        for j in mol2:
            tani_ecfc4=DataStructs.TanimotoSimilarity(fe1[i],fe2[j])
            tani_maccs=DataStructs.FingerprintSimilarity(fm1[i],fm2[j], metric=DataStructs.TanimotoSimilarity)
            fo.write(i+'\t'+j+'\t'+str(tani_ecfc4)+'\t'+str(tani_maccs)+'\n')
    fo.close()

fe1,fm1,mol1 = finger(edcs1,'user')
print ('Hello')
fe2,fm2,mol2 = finger(edcs2,'drugbank')
print ('World')
tc(fe1,fm1,fe2,fm2,mol1,mol2, 'IMPPAT2_DrugBankApprv_CSN.txt')
#tc(fe1,fm1,fe2,fm2,mol1,mol2, '14_APPRV_tc-2.txt')


