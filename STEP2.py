import random
import numpy as np
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

import matplotlib.pyplot as plt
import seaborn as sns

# ===================== REACTIONS =====================

ester_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2][C:3](=O)O.[O:4][C:5]>>[C:1]=[C:2][C:3](=O)[O:4][C:5]"
)

amide_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2][C:3](=O)O.[N:4][C:5]>>[C:1]=[C:2][C:3](=O)[N:4][C:5]"
)

methacrylate_rxn = AllChem.ReactionFromSmarts(
    "[C:1]=[C:2]C(=O)O.[O:3][C:4]>>[C:1]=[C:2]C(=O)[O:3][C:4]"
)

# ===================== BASE MOLECULES =====================

acrylic_acid = Chem.MolFromSmiles("C=CC(=O)O")
methacrylic_acid = Chem.MolFromSmiles("CC(=C)C(=O)O")

# Alcohols
alcohols = [Chem.MolFromSmiles(s) for s in [
    "CO","CCO","CCCO","CCCCO",
    "OCCO","OCCCO","OCC(O)CO","OCC(O)C(O)CO",
    "CC(C)O","CC(C)(C)O",
    "c1ccccc1O"   # phenol
]]

# Amines
amines = [Chem.MolFromSmiles(s) for s in [
    "CN","CCN","CCCN",
    "NCCO","NCCCO",
    "NC(C)C",
    "c1ccccc1N"   # aniline
]]

# ===================== AROMATICS =====================

aromatic_alcohols = [
    "c1ccccc1O",
    "c1ccccc1CO",
    "Oc1ccc(O)cc1",
    "COc1ccccc1"
]

aromatic_amines = [
    "c1ccccc1N",
    "Nc1ccc(N)cc1",
    "CNc1ccccc1",
    "c1ccncc1N"
]

alcohols += [Chem.MolFromSmiles(s) for s in aromatic_alcohols]
amines   += [Chem.MolFromSmiles(s) for s in aromatic_amines]

# ===================== REACTION RUNNER =====================

def run_reaction(rxn, reactant1, reactant_list):
    products = []
    for r2 in reactant_list:
        try:
            ps = rxn.RunReactants((reactant1, r2))
            for p in ps:
                products.append(Chem.MolToSmiles(p[0], canonical=True))
        except:
            continue
    return products

# ===================== GENERATION =====================

def generate_monomers_expanded():
    smiles_set = set()

    smiles_set.update(run_reaction(ester_rxn, acrylic_acid, alcohols))
    smiles_set.update(run_reaction(amide_rxn, acrylic_acid, amines))
    smiles_set.update(run_reaction(methacrylate_rxn, methacrylic_acid, alcohols))
    smiles_set.update(run_reaction(amide_rxn, methacrylic_acid, amines))

    # Aromatic grafted monomers
    smiles_set.update([
        "C=CC(=O)Oc1ccccc1",
        "C=CC(=O)Nc1ccccc1",
        "CC(=C)C(=O)Oc1ccccc1",
        "CC(=C)C(=O)Nc1ccccc1"
    ])

    return list(smiles_set)

# ===================== VALIDATION =====================

def validate_smiles(smiles_list):
    valid = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid.append(Chem.MolToSmiles(mol))
    return list(set(valid))

# ===================== DIVERSITY FILTER =====================

def diversify_smiles(smiles_list, threshold=0.8):
    diverse, fps = [], []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)

        if all(TanimotoSimilarity(fp, f) < threshold for f in fps):
            diverse.append(s)
            fps.append(fp)

    return diverse

# ===================== AUGMENTATION =====================

def augment_smiles(smiles_list, target_size=120):
    new_smiles = set(smiles_list)
    
    bases = ["C","CC","CCC","CCCC","CCCCC","CCCCCC","CCCCCCC"]
    groups = ["O","N","CO","CN","OO","NO","C=O","C#N"]
    aromatics = ["c1ccccc1","c1ccncc1","c1ccc(O)cc1","COc1ccccc1","c1ccccc1C(=O)O","c1ccccc1C=O"]

    attempts = 0
    max_attempts = 2500   # safety limit

    while len(new_smiles) < target_size and attempts < max_attempts:
        attempts += 1

        if random.random() < 0.4:
            s = random.choice(aromatics) + random.choice(groups)
        else:
            s = random.choice(bases) + random.choice(groups)

        mol = Chem.MolFromSmiles(s)
        if mol:
            new_smiles.add(Chem.MolToSmiles(mol))

    print("Augmentation attempts:", attempts)
    return list(new_smiles)

# ===================== PIPELINE =====================

smiles_data = generate_monomers_expanded()
smiles_data = validate_smiles(smiles_data)
smiles_data = diversify_smiles(smiles_data)

smiles_data = augment_smiles(smiles_data, 120)
smiles_data = validate_smiles(smiles_data)
smiles_data = diversify_smiles(smiles_data)

df = pd.DataFrame({"SMILES": smiles_data})

print("Final dataset size:", len(df))

# ===================== FEATURES =====================

n = len(df)

df["chain_length"] = np.random.randint(50, 300, n)
df["crosslink_density"] = np.random.uniform(0.01, 0.3, n)
df["polymer_concentration"] = np.random.uniform(0.2, 0.9, n)
df["temperature"] = np.random.uniform(273, 350, n)
df["pH"] = np.random.uniform(4, 9, n)

df["LogP"] = df["SMILES"].apply(
    lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s))
)



# ===================== PROPERTIES =====================

def generate_properties(row):
    chi = 0.5 + 0.1 * row["LogP"]
    Q = (1/(row["crosslink_density"]+1e-3))*np.exp(-chi)*(1+0.5*row["polymer_concentration"])
    xi = np.sqrt(Q)/(row["crosslink_density"]+1e-3)

    return pd.Series({
        "swelling_ratio": Q,
        "diffusion_coefficient": xi/(row["chain_length"]+1),
        "elastic_modulus": row["crosslink_density"]*row["chain_length"]*(1+row["polymer_concentration"]),
        "porosity": Q/(Q+1),
        "degradation_rate": 1/(row["chain_length"]*row["crosslink_density"]+1)
    })

df = pd.concat([df, df.apply(generate_properties, axis=1)], axis=1)

# ===================== CHEM FEATURES =====================

def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return pd.Series()

    return pd.Series({
        # ================= BASIC =================
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),

        # ================= STRUCTURE =================
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),

        # ================= H-BOND =================
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),

        # ================= POLARITY =================
        "TPSA": Descriptors.TPSA(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),

        # ================= TOPOLOGICAL =================
        "Chi0": rdMolDescriptors.CalcChi0n(mol),
        "Chi1": rdMolDescriptors.CalcChi1n(mol),
        "Chi2": rdMolDescriptors.CalcChi2n(mol),
        "Chi3": rdMolDescriptors.CalcChi3n(mol),
        "Chi4": rdMolDescriptors.CalcChi4n(mol),

        # ================= SHAPE =================
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),

        # ================= COMPLEXITY =================
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),

        # ================= CHARGES =================
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),

        # ================= ATOM INFO =================
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),

        # ================= RING DETAILS =================
        "NumAromaticAtoms": sum([a.GetIsAromatic() for a in mol.GetAtoms()]),
        "IsAromatic": int(any(a.GetIsAromatic() for a in mol.GetAtoms())),

        # ================= SURFACE / SIZE =================
        "MolMR": Descriptors.MolMR(mol),

        # ================= DRUG-LIKE =================
        "qed": Descriptors.qed(mol),

        # ================= ADDITIONAL STRONG FEATURES =================
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),

        "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),

        "NumLipinskiHBA": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "NumLipinskiHBD": rdMolDescriptors.CalcNumLipinskiHBD(mol),

        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),

        # ================= ELECTRONIC =================
        "MaxAbsPartialCharge": max(
            abs(Descriptors.MaxPartialCharge(mol)),
            abs(Descriptors.MinPartialCharge(mol))
        ),

        # ================= FLEXIBILITY =================
        "RotatableBondFraction": (
            Descriptors.NumRotatableBonds(mol) / (mol.GetNumHeavyAtoms() + 1)
        )
    })
    
chem_features = df["SMILES"].apply(compute_features)
df = pd.concat([df, chem_features], axis=1)

# ===================== FINGERPRINT =====================

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    return fp

df["fingerprint"] = df["SMILES"].apply(get_fp)

# ===================== SIMILARITY =====================

n = len(df)
sim_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(i,n):
        sim = TanimotoSimilarity(df["fingerprint"][i], df["fingerprint"][j])
        sim_matrix[i][j]=sim
        sim_matrix[j][i]=sim

print("Average similarity:", np.mean(sim_matrix))

# ===================== SAVE =====================


df.to_excel("hydrogel_curated_final.xlsx", index=False)
print("Done!")
