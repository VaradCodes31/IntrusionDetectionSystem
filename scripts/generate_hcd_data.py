import pandas as pd
import numpy as np

# Load our research seed with 12 classes
seed_df = pd.read_csv('research_seed.csv')
seed_df.columns = seed_df.columns.str.strip()

# Partition classes
quantum_classes = ['BENIGN', 'DoS Hulk', 'PortScan']
classical_only_classes = [c for c in seed_df['Label'].unique() if c not in quantum_classes]

print(f"Quantum Overlap Classes: {quantum_classes}")
print(f"Classical-Only Classes: {classical_only_classes}")

data = []

def generate_samples(subset_df, target_count, category_label):
    subset_df = subset_df.reset_index(drop=True)
    features = subset_df.drop('Label', axis=1)
    labels_col = subset_df['Label']
    
    for i in range(target_count):
        idx = np.random.randint(0, len(subset_df))
        base_row = features.iloc[idx].copy()
        label = labels_col.iloc[idx]
        
        # Inject noise for synthetic expansion
        for col in features.columns:
            val = base_row[col]
            # Handle mixed objects/numeric
            try:
                val = float(val)
                noise = np.random.normal(0, abs(val) * 0.1 + 0.5)
                base_row[col] = val + noise
            except:
                pass
                
        row_dict = base_row.to_dict()
        row_dict["Label"] = label
        row_dict["Research_Cluster"] = category_label
        data.append(row_dict)

# 1. 60% Overlap
overlap_seed = seed_df[seed_df['Label'].isin(quantum_classes)]
generate_samples(overlap_seed, 1200, "OVERLAP")

# 2. 40% Classical-Only
classical_seed = seed_df[seed_df['Label'].isin(classical_only_classes)]
generate_samples(classical_seed, 800, "CLASSICAL_ONLY")

df = pd.DataFrame(data)
# Final cleaning
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].clip(lower=0).fillna(0)

# Save
df.to_csv("varied_traffic.csv", index=False)
print(f"Created varied_traffic.csv with {len(df)} samples.")
print(df['Label'].value_counts())
