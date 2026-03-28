import random
import numpy as np
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

import matplotlib.pyplot as plt
import seaborn as sns

# Esterification: acrylic acid + alcohol → acrylate
ester_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2][C:3](=O)O.[O:4][C:5]>>[C:1]=[C:2][C:3](=O)[O:4][C:5]"
)

# Amide formation: acrylic acid + amine → acrylamide
amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2][C:3](=O)O.[N:4][C:5]>>[C:1]=[C:2][C:3](=O)[N:4][C:5]"
)

methacrylic_acid = Chem.MolFromSmiles("CC(=C)C(=O)O")

methacrylate_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2]C(=O)O.[O:3][C:4]>>[C:1]=[C:2]C(=O)[O:3][C:4]"
)

vinyl_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2].[O:3][C:4]>>[C:1]=[C:2][O:3][C:4]"
)

# Acrylic backbone
acrylic_acid = Chem.MolFromSmiles("C=CC(=O)O")

# Expanded alcohols
alcohols = [Chem.MolFromSmiles(s) for s in [
    "CO", "CCO", "CCCO", "CCCCO",
    "OCCO", "OCCCO", "OCC(O)CO",
    "OCC(O)C(O)CO",          # glycerol-like
    "CC(C)O", "CC(C)(C)O",   # branched
    "c1ccccc1O"              # phenol
]]
alcohols.extend([
    Chem.MolFromSmiles(s) for s in [
        "OCCOCCO",
        "OCC(O)COCCO",
        "OCCNCCO"
    ]
])

# Expanded amines
amines = [Chem.MolFromSmiles(s) for s in [
    "CN", "CCN", "CCCN",
    "NCCO", "NCCCO",
    "NC(C)C",                # branched
    "c1ccccc1N"              # aniline
]]
amines.extend([
    Chem.MolFromSmiles(s) for s in [
        "NCCN",
        "NCCCN",
        "NC(C)CN"
    ]
])

def run_reaction(rxn, reactant1, reactant_list):
    products = []
    for r2 in reactant_list:
        try:
            ps = rxn.RunReactants((reactant1, r2))
            for p in ps:
                smiles = Chem.MolToSmiles(p[0], canonical=True)
                products.append(smiles)
        except:
            continue
    return products
    

def generate_monomers_expanded():
    smiles_set = set()
    
    # Existing
    smiles_set.update(run_reaction(ester_rxn, acrylic_acid, alcohols))
    smiles_set.update(run_reaction(amide_rxn, acrylic_acid, amines))
    
    # NEW additions
    smiles_set.update(run_reaction(methacrylate_rxn, methacrylic_acid, alcohols))
    smiles_set.update(run_reaction(amide_rxn, methacrylic_acid, amines))
    
    return list(smiles_set)
    
def validate_smiles(smiles_list):
    valid = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol))
    return list(set(valid))

smiles_data = generate_monomers_expanded()
smiles_data = validate_smiles(smiles_data)

df = pd.DataFrame({"SMILES": smiles_data})

df = df.loc[df.index.repeat(2)].reset_index(drop=True)

extra = df.sample(12, random_state=42)
df = pd.concat([df, extra]).reset_index(drop=True)

print("Total samples", len(df))

n = len(df)

df["chain_length"] = np.random.randint(50, 300, n)
df["crosslink_density"] = np.random.uniform(0.01, 0.3, n)
df["polymer_concentration"] = np.random.uniform(0.2, 0.9, n)
df["temperature"] = np.random.uniform(273, 350, n)
df["pH"] = np.random.uniform(4, 9, n)

df["LogP"] = df["SMILES"].apply(
    lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s))
)

def generate_properties(row):
    
    crosslink = row["crosslink_density"]
    chain = row["chain_length"]
    conc = row["polymer_concentration"]
    temp = row["temperature"]
    logp = row["LogP"]
    
    # Polymer-solvent interaction
    chi = 0.5 + 0.1 * logp
    
    # Swelling ratio
    Q = (1 / (crosslink + 1e-3)) * np.exp(-chi) * (1 + 0.5 * conc)
    
    # Mesh size
    xi = np.sqrt(Q) / (crosslink + 1e-3)
    
    # Diffusion
    D = xi / (chain + 1)
    
    # Modulus
    E = crosslink * chain * (1 + conc)
    
    # Porosity
    porosity = Q / (Q + 1)
    
    # Degradation
    degradation = 1 / (chain * crosslink + 1)
    
    return pd.Series({
        "swelling_ratio": Q,
        "diffusion_coefficient": D,
        "elastic_modulus": E,
        "porosity": porosity,
        "degradation_rate": degradation
    })
    
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return pd.Series()
    
    return pd.Series({
        # Basic
        "MolWt": Descriptors.MolWt(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        
        # Structure
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        
        # H-bonding
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        
        # Surface / polarity
        "TPSA": Descriptors.TPSA(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        
        # Topological indices
        "Chi0": rdMolDescriptors.CalcChi0n(mol),
        "Chi1": rdMolDescriptors.CalcChi1n(mol),
        "Chi2": rdMolDescriptors.CalcChi2n(mol),
        
        # Kappa shape indices
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        
        # Complexity
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        
        # Additional useful ones
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
        
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
        
        "BalabanJ": Descriptors.BalabanJ(mol),
        
        "BertzCT": Descriptors.BertzCT(mol),
        
        "Ipc": Descriptors.Ipc(mol),
        
        "MolMR": Descriptors.MolMR(mol),
        
        "qed": Descriptors.qed(mol)
    })
    
props = df.apply(generate_properties, axis=1)
df = pd.concat([df.drop(columns=[
    "swelling_ratio",
    "diffusion_coefficient",
    "elastic_modulus",
    "porosity",
    "degradation_rate"
], errors="ignore"), props], axis=1)

feature_df = df["SMILES"].apply(compute_features)

df = pd.concat([df, feature_df], axis=1)

df = df.drop_duplicates()
print("Before dropna:", len(df))
df = df.dropna()
print("After dropna:", len(df))
df = df.reset_index(drop=True)

def get_name(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
        r = requests.get(url).json()
        return r["PropertyTable"]["Properties"][0]["IUPACName"]
    except:
        return "Unknown"

df["IUPAC_Name"] = df["SMILES"].apply(get_name)

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Create generator once
morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return morgan_gen.GetFingerprint(mol)
    
df["fingerprint"] = df["SMILES"].apply(get_fingerprint)

from rdkit.DataStructs import TanimotoSimilarity

sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim_matrix[i][j] = TanimotoSimilarity(
            df["fingerprint"][i],
            df["fingerprint"][j]
        )



df["swelling_ratio"] = df["swelling_ratio"].clip(0, 50)
df["diffusion_coefficient"] = df["diffusion_coefficient"].clip(0, 2)
df["Ipc"] = df["Ipc"].clip(0, 1e6)

print(df.describe())

print("Average similarity:", np.mean(sim_matrix))

summary = df.describe(include='all').T
summary.to_excel("hydrogel_summary.xlsx",index=False)

corr = df.corr(numeric_only=True)
corr.to_excel("correlation_matrix.xlsx",index=False)

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.show()

df.hist(figsize=(12,10))
plt.show()

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("hydrogel_curated_final.csv", index=False)
print("1 Done!")
df.to_excel("hydrogel_curated.xlsx", index=False)
print("2 Done!")
