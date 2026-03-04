import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_features(df):
    print("Preparing features and labels...")

    # Separate features and target
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop rows that became NaN after conversion
    X = X.dropna()
    y = y.loc[X.index]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Labels encoded.")
    print("Number of classes:", len(le.classes_))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Train-test split completed.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, le