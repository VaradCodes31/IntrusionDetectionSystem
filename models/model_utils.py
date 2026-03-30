import joblib
import os

def save_object(obj, filename):
    """Saves a python object to the models directory."""
    path = os.path.join("models", filename)
    joblib.dump(obj, path)
    print(f"Object saved to {path}")

def load_object(filename):
    """Loads a python object from the models directory."""
    path = os.path.join("models", filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None
