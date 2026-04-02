import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('varied_traffic.csv')

# Drop non-feature columns
X = df.drop(['Label', 'Research_Cluster'], axis=1, errors='ignore')

# Handle infinite or overly large values for float32 compatibility
X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
X = X.replace([np.inf, -np.inf], 1.0e15) # Cap large values
X = np.clip(X, -1.0e15, 1.0e15) # Clip to a stable float32 range

y = df['Label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Training RandomForest (12-class Baseline)...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=1
)
model.fit(X, y_encoded)

joblib.dump(model, 'models/xgboost_model.pkl')
print(f"RandomForest Baseline saved. Classes: {le.classes_}")
