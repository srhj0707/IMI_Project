import pandas as pd
import random

# ✅ Load dataset
df = pd.read_excel("hydrogel_100_samples_final.xlsx")

# -----------------------------
# ✅ Helper classifications
# -----------------------------
hydrophilic = [
    "acrylamide", "acrylic acid", "methacrylic acid",
    "vinyl alcohol", "ethylene glycol",
    "N-vinyl pyrrolidone", "itaconic acid", "maleic acid"
]

biodegradable = [
    "chitosan", "gelatin", "vinyl alcohol", "acrylic acid"
]

# -----------------------------
# ✅ Feature generation functions
# -----------------------------

def crosslinker_type():
    return random.choice(["MBAA", "EGDMA", "Glutaraldehyde", "None"])

def crosslinker_conc():
    return round(random.uniform(0.1, 5.0), 2)  # %

def mesh_size():
    return round(random.uniform(5, 100), 2)  # nm

def water_retention(swelling):
    return round(swelling * random.uniform(0.6, 0.9), 2)

def enthalpy_change():
    return round(random.uniform(-50, 10), 2)  # kJ/mol

def osmotic_pressure():
    return round(random.uniform(0.5, 10), 2)  # atm

def biodegradability(m1, m2):
    if m1 in biodegradable or m2 in biodegradable:
        return random.choice(["High", "Medium"])
    return random.choice(["Low", "Medium"])

def degradation_half_life(bio):
    if bio == "High":
        return random.randint(5, 20)
    elif bio == "Medium":
        return random.randint(20, 60)
    else:
        return random.randint(60, 180)

def toxicity_index(m1, m2):
    if "acrylonitrile" in [m1, m2]:
        return round(random.uniform(0.6, 1.0), 2)
    return round(random.uniform(0.1, 0.5), 2)

def adsorption_capacity():
    return round(random.uniform(10, 200), 2)  # mg/g

def absorption_rate():
    return round(random.uniform(0.1, 5.0), 2)  # g/g/min

def contact_angle(m1, m2):
    if m1 in hydrophilic or m2 in hydrophilic:
        return round(random.uniform(20, 60), 2)
    return round(random.uniform(60, 110), 2)

# -----------------------------
# ✅ Apply features to dataset
# -----------------------------

# Crosslinker
df["Crosslinker Type"] = [crosslinker_type() for _ in range(len(df))]
df["Crosslinker Concentration (%)"] = [crosslinker_conc() for _ in range(len(df))]

# Mesh size
df["Mesh Size (nm)"] = [mesh_size() for _ in range(len(df))]

# Water retention (based on swelling)
df["Water Retention Capacity"] = df["Swelling_Ratio"].apply(lambda x: water_retention(x))

# Thermodynamic
df["Enthalpy Change (kJ/mol)"] = [enthalpy_change() for _ in range(len(df))]
df["Osmotic Pressure (atm)"] = [osmotic_pressure() for _ in range(len(df))]

# Biodegradation
df["Biodegradability"] = df.apply(lambda row: biodegradability(row["Monomer_A"], row["Monomer_B"]), axis=1)
df["Degradation Half-life (days)"] = df["Biodegradability"].apply(degradation_half_life)

# Toxicity
df["Toxicity Index"] = df.apply(lambda row: toxicity_index(row["Monomer_A"], row["Monomer_B"]), axis=1)

# Adsorption & absorption
df["Adsorption Capacity (mg/g)"] = [adsorption_capacity() for _ in range(len(df))]
df["Absorption Rate (g/g/min)"] = [absorption_rate() for _ in range(len(df))]

# Surface property
df["Contact Angle (deg)"] = df.apply(lambda row: contact_angle(row["Monomer_A"], row["Monomer_B"]), axis=1)

# -----------------------------
# ✅ Save final dataset
# -----------------------------
df.to_excel("hydrogel_complete_dataset.xlsx", index=False)

print("✅ ALL features added successfully! Final dataset ready.")
