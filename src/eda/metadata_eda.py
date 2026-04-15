import pandas as pd

file_path = "data/BraTS-PTG supplementary demographic information and metadata.xlsx" 

df = pd.read_excel(file_path)
print(df.head())
print("\n")

print("INFORMACIÓN GENERAL:")
df.info()
print("\n")

print("ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())