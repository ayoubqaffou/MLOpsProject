import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def train():
    project_root = Path(__file__).resolve().parents[2]

    df = pd.read_csv(project_root / "data" / "raw" / "student_placement_data_v1.csv")

    X = df.drop("Placement", axis=1)
    Y = df["Placement"]

    model = RandomForestClassifier()
    model.fit(X, Y)

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.pkl")

    print("Model trained on student placement dataset!")

if __name__ == "__main__":
    train()