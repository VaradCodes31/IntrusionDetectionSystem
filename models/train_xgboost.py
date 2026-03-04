from xgboost import XGBClassifier
import joblib


def train_model(X_train, y_train):
    print("Starting XGBoost training...")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(set(y_train)),
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training completed.")
    return model


def save_model(model):
    path = "models/xgboost_model.pkl"
    joblib.dump(model, path)
    print(f"Model saved to {path}")