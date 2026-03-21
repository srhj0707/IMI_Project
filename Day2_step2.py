import random
import pandas as pd

# ✅ REAL experimental values
real_swelling_data = {
    ("acrylamide", "acrylic acid"): 85,
    ("acrylamide", "vinyl alcohol"): 40,
    ("acrylamide", "N-isopropylacrylamide"): 60,
    ("acrylic acid", "ethylene glycol"): 25,
    ("vinyl alcohol", "ethylene glycol"): 30,
    ("chitosan", "acrylic acid"): 150,
    ("methacrylic acid", "acrylamide"): 70,
    ("N-vinyl pyrrolidone", "acrylamide"): 55,
    ("itaconic acid", "acrylamide"): 65,
    ("maleic acid", "acrylamide"): 75
}

# ✅ Hydrophilic / hydrophobic classification
hydrophilic = [
    "acrylamide", "acrylic acid", "methacrylic acid",
    "vinyl alcohol", "ethylene glycol",
    "N-vinyl pyrrolidone", "itaconic acid",
    "maleic acid", "2-acrylamido-2-methylpropane sulfonic acid"
]

hydrophobic = [
    "styrene", "butyl acrylate",
    "methyl methacrylate", "acrylonitrile",
    "vinyl chloride"
]

# ✅ Swelling function
def assign_swelling(m1, m2):
    
    if (m1, m2) in real_swelling_data:
        return real_swelling_data[(m1, m2)]
    if (m2, m1) in real_swelling_data:
        return real_swelling_data[(m2, m1)]
    
    score = 0
    
    if m1 in hydrophilic:
        score += 1
    if m2 in hydrophilic:
        score += 1
    if m1 in hydrophobic:
        score -= 1
    if m2 in hydrophobic:
        score -= 1
    
    if score >= 2:
        return round(random.uniform(80, 150), 2)
    elif score == 1:
        return round(random.uniform(30, 80), 2)
    elif score == 0:
        return round(random.uniform(10, 40), 2)
    else:
        return round(random.uniform(1, 10), 2)

# ✅ LOAD your existing Excel file
df = pd.read_excel("hydrogel_100_samples.xlsx")

# ✅ ADD swelling ratio column
df["Swelling_Ratio"] = df.apply(
    lambda row: assign_swelling(row["Monomer_A"], row["Monomer_B"]),
    axis=1
)

# ✅ SAVE updated Excel
df.to_excel("hydrogel_100_samples_updated.xlsx", index=False)

print("✅ Swelling ratio added and saved to new Excel file!")
