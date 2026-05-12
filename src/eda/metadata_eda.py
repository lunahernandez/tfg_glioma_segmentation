import pandas as pd


file_path = "data/BraTS-PTG supplementary demographic information and metadata.xlsx"

df = pd.read_excel(file_path)
print(df.head())
print("\n")

print("GENERAL INFORMATION:")
df.info()
print("\n")

print("DESCRIPTIVE STATISTICS:")
print(df.describe())
