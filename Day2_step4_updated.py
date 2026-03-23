import pandas as pd
import random
import pubchempy as pcp

# -----------------------------
# ✅ STEP 1: Load dataset
# -----------------------------
df = pd.read_excel("hydrogel_100_samples_final.xlsx")

# -----------------------------
# ✅ STEP 2: PubChem API function
# -----------------------------
def get_pubchem_data(name):
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            return {
                "MW": c.molecular_weight,
                "LogP": c.xlogp,
                "H_donors": c.h_bond_donor_count,
                "H_acceptors": c.h_bond_acceptor_count,
                "TPSA": c.tpsa
            }
    except:
        pass

    return {
        "MW": None,
        "LogP": None,
        "H_donors": None,
        "H_acceptors": None,
        "TPSA": None
    }

# -----------------------------
# ✅ STEP 3: Extract API features
# -----------------------------
features_A = df["Monomer_A"].apply(get_pubchem_data).apply(pd.Series)
features_B = df["Monomer_B"].apply(get_pubchem_data).apply(pd.Series)

features_A.columns = [col + "_A" for col in features_A.columns]
features_B.columns = [col + "_B" for col in features_B.columns]

df = pd.concat([df, features_A, features_B], axis=1)

# -----------------------------
# ✅ STEP 4: Fill missing values
# -----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# ✅ STEP 5: Crosslinker features
# -----------------------------
def crosslinker_type():
    return random.choice(["MBAA", "EGDMA", "Glutaraldehyde", "None"])

def crosslinker_conc():
    return round(random.uniform(0.1, 5.0), 2)

df["Crosslinker Type"] = [crosslinker_type() for _ in range(len(df))]
df["Crosslinker Concentration (%)"] = [crosslinker_conc() for _ in range(len(df))]

# -----------------------------
# ✅ STEP 6: Derived chemical features
# -----------------------------
df["Hydrophilicity_Index"] = (
    (df["H_donors_A"] + df["H_donors_B"]) +
    (df["H_acceptors_A"] + df["H_acceptors_B"]) -
    (df["LogP_A"] + df["LogP_B"])
)

# -----------------------------
# ✅ STEP 7: Crosslinking Density
# -----------------------------
df["Crosslinking_Density"] = df["Crosslinker Concentration (%)"] * 0.1

# -----------------------------
# ✅ STEP 8: Mesh Size (correlated)
# -----------------------------
df["Mesh Size (nm)"] = 100 / (df["Crosslinking_Density"] + 1)

# -----------------------------
# ✅ STEP 9: Predicted Swelling
# -----------------------------
df["Predicted_Swelling"] = (
    df["Hydrophilicity_Index"] * 2
) / (df["Crosslinking_Density"] + 1)

# -----------------------------
# ✅ STEP 10: Water Retention
# -----------------------------
df["Water Retention Capacity"] = df["Predicted_Swelling"] * 0.8

# -----------------------------
# ✅ STEP 11: Contact Angle
# -----------------------------
df["Contact Angle (deg)"] = 110 - df["Hydrophilicity_Index"] * 2
df["Contact Angle (deg)"] = df["Contact Angle (deg)"].clip(20, 110)

# -----------------------------
# ✅ STEP 12: Biodegradability
# -----------------------------
df["Biodegradability Score"] = (
    df["TPSA_A"] + df["TPSA_B"]
) / 50

df["Biodegradability"] = pd.cut(
    df["Biodegradability Score"],
    bins=[-1, 1, 3, 10],
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# ✅ STEP 13: Degradation Half-life
# -----------------------------
def degradation_half_life(level):
    if level == "High":
        return random.randint(5, 20)
    elif level == "Medium":
        return random.randint(20, 60)
    return random.randint(60, 150)

df["Degradation Half-life (days)"] = df["Biodegradability"].apply(degradation_half_life)

# -----------------------------
# ✅ STEP 14: Adsorption Capacity
# -----------------------------
df["Adsorption Capacity (mg/g)"] = (
    df["TPSA_A"] + df["TPSA_B"]
) * 2

# -----------------------------
# ✅ STEP 15: Absorption Rate
# -----------------------------
df["Absorption Rate (g/g/min)"] = (
    df["Predicted_Swelling"] / df["Mesh Size (nm)"]
)

# -----------------------------
# ✅ STEP 16: Osmotic Pressure (linked)
# -----------------------------
df["Osmotic Pressure (atm)"] = (
    df["Predicted_Swelling"] * 0.05
)

# -----------------------------
# ✅ STEP 17: Enthalpy Change (semi-correlated)
# -----------------------------
df["Enthalpy Change (kJ/mol)"] = -df["Hydrophilicity_Index"] * 2

# -----------------------------
# ✅ STEP 18: Toxicity Index
# -----------------------------
def toxicity_index(m1, m2):
    if "acrylonitrile" in str(m1).lower() or "acrylonitrile" in str(m2).lower():
        return random.uniform(0.6, 1.0)
    return random.uniform(0.1, 0.5)

df["Toxicity Index"] = df.apply(
    lambda row: toxicity_index(row["Monomer_A"], row["Monomer_B"]),
    axis=1
)

# -----------------------------
# ✅ STEP 19: Final cleanup
# -----------------------------
df.replace([float('inf'), -float('inf')], 0, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# ✅ STEP 20: Save dataset
# -----------------------------
df.to_excel("hydrogel_complete_dataset.xlsx", index=False)

print("✅ Final dataset with API + correlated features ready!")
