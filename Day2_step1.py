monomers = [
    "acrylamide",
    "acrylic acid",
    "methacrylic acid",
    "N-isopropylacrylamide",
    "hydroxyethyl methacrylate",
    "ethylene glycol",
    "vinyl alcohol",
    "N-vinyl pyrrolidone",
    "2-acrylamido-2-methylpropane sulfonic acid",
    "itaconic acid",
    "maleic acid",
    "vinyl acetate",
    "styrene sulfonic acid",
    "allylamine",
    "diethylaminoethyl methacrylate",
    "butyl acrylate",
    "methyl methacrylate",
    "acrylonitrile",
    "vinyl chloride",
    "ethylene"
]

import itertools
import random

pairs = list(itertools.combinations(monomers, 2))

samples = []

ratios = ["50:50", "60:40", "70:30", "80:20"]

for i in range(100):
    m1, m2 = random.choice(pairs)
    ratio = random.choice(ratios)
    
    samples.append({
        "Monomer_A": m1,
        "Monomer_B": m2,
        "Ratio": ratio
    })
    

import pubchempy as pcp
import time

def get_data(name):
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            return c.molecular_weight, c.connectivity_smiles
    except:
        return None, None
    
    return None, None


final_data = []

for i, s in enumerate(samples):
    print(f"Processing {i+1}/100")
    
    mw1, sm1 = get_data(s["Monomer_A"])
    mw2, sm2 = get_data(s["Monomer_B"])
    
    final_data.append({
        "Monomer_A": s["Monomer_A"],
        "Monomer_B": s["Monomer_B"],
        
        # Convert ratio to numeric (VERY IMPORTANT for ML)
        "Ratio_A": int(s["Ratio"].split(":")[0]) / 100,
        "Ratio_B": int(s["Ratio"].split(":")[1]) / 100,
        
        "MW_A": mw1,
        "MW_B": mw2,
        
        "SMILES_A": sm1,
        "SMILES_B": sm2,
        
        # Fill later from literature / MatWeb
        "Density": None,
        "Tg": None,
        
        "Polymer_Type": "Hydrogel"
    })
    
    time.sleep(0.5)
    
import pandas as pd

df = pd.DataFrame(final_data)
df.to_excel("hydrogel_100_samples.xlsx", index=False)

print("✅ Your 100-sample dataset is ready!")
