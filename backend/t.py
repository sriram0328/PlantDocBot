# In Python console or notebook
import pandas as pd

# Load your parquet file
df = pd.read_parquet("plant_disease_captions.parquet")

# Check the disease_label column
print(df['disease_label'].unique())

# Check if "Tomato Late_blight" or similar exists
print(df[df['disease_label'].str.contains('Late', case=False, na=False)])