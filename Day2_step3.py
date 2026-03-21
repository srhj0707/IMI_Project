import pandas as pd

# ✅ REAL / semi-real values (from materials data sources like MatWeb)
real_properties = {
    "acrylamide": {"Density": 1.13, "Tg": 165},
    "acrylic acid": {"Density": 1.05, "Tg": 106},
    "methacrylic acid": {"Density": 1.02, "Tg": 160},
    "N-isopropylacrylamide": {"Density": 1.10, "Tg": 140},
    "hydroxyethyl methacrylate": {"Density": 1.07, "Tg": 55},
    "ethylene glycol": {"Density": 1.11, "Tg": -12},
    "vinyl alcohol": {"Density": 1.19, "Tg": 85},
    "N-vinyl pyrrolidone": {"Density": 1.03, "Tg": 175},
    "itaconic acid": {"Density": 1.63, "Tg": 130},
    "maleic acid": {"Density": 1.59, "Tg": 135},
    "vinyl acetate": {"Density": 0.93, "Tg": 30},
    "styrene sulfonic acid": {"Density": 1.20, "Tg": 180},
    "allylamine": {"Density": 0.76, "Tg": -50},
    "diethylaminoethyl methacrylate": {"Density": 0.92, "Tg": 60},
    "butyl acrylate": {"Density": 0.90, "Tg": -54},
    "methyl methacrylate": {"Density": 0.94, "Tg": 105},
    "acrylonitrile": {"Density": 0.81, "Tg": 95},
    "vinyl chloride": {"Density": 1.40, "Tg": 80},
    "ethylene": {"Density": 0.97, "Tg": -125}
}

# ✅ Function to fetch values
def get_properties(monomer):
    return real_properties.get(monomer, {"Density": None, "Tg": None})

# ✅ Load your Excel file
df = pd.read_excel("hydrogel_100_samples_updated.xlsx")

# ✅ Create new columns for each monomer
df["Density_A"] = df["Monomer_A"].apply(lambda m: get_properties(m)["Density"])
df["Density_B"] = df["Monomer_B"].apply(lambda m: get_properties(m)["Density"])

df["Tg_A"] = df["Monomer_A"].apply(lambda m: get_properties(m)["Tg"])
df["Tg_B"] = df["Monomer_B"].apply(lambda m: get_properties(m)["Tg"])

# ✅ Save updated file
df.to_excel("hydrogel_100_samples_final.xlsx", index=False)

print("✅ Density and Tg values added successfully!")
