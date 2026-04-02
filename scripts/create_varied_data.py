import pandas as pd
import numpy as np

# Load local sample rows to create noisy replicas with realistic signatures
sample_df = pd.read_csv('sample_test.csv')
sample_df.columns = sample_df.columns.str.strip() # Normalize headers

# Ensure only labels supported by the retrained model are generated
# Based on retraining: {'BENIGN': 0, 'DoS Hulk': 1, 'PortScan': 2}
labels = ['BENIGN', 'DoS Hulk', 'PortScan']
sample_df = sample_df[sample_df['Label'].isin(labels)]

features = sample_df.drop('Label', axis=1)
labels_col = sample_df['Label']

data = []

def generate_diverse_samples(num_samples=100):
    for i in range(num_samples):
        # Pick a random sample index
        idx = np.random.randint(0, len(sample_df))
        base_row = features.iloc[idx].copy()
        label = labels_col.iloc[idx]
        
        # Add 10% Gaussian noise to each numeric feature to create more variety
        for col in features.columns:
            val = base_row[col]
            if pd.api.types.is_numeric_dtype(type(val)):
                # Inject slight non-zero value if it was zero
                if val == 0:
                    val = np.random.uniform(0.1, 5)
                
                noise = np.random.normal(0, abs(val) * 0.1 + 0.5) 
                base_row[col] = val + noise
                
        row_dict = base_row.to_dict()
        row_dict["Label"] = label
        data.append(row_dict)

generate_diverse_samples(2000)
df = pd.DataFrame(data)

# Final clipping to ensure no negative values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].clip(lower=0)

df.to_csv("varied_traffic.csv", index=False)
print(f"Created varied_traffic.csv with {len(df)} diverse noisy samples. Headers are clean.")
