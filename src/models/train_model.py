import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train(n_estimators: int = 100, random_state: int = 42):
    project_root = Path(__file__).resolve().parents[2]

    df = pd.read_csv(project_root / "data" / "raw" / "student_placement_data_v1.csv")

    X = df.drop("Placement", axis=1)
    Y = df["Placement"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    mlflow_db_path = str(project_root / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

    mlflow.set_experiment("Student Placement Prediction")

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_seed{random_state}"):
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", accuracy)

        # models_dir = project_root / "models"
        # models_dir.mkdir(parents=True, exist_ok=True)
        # joblib.dump(model, models_dir / "model.pkl")

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=['sklearn.tree._tree.Tree'])

        print(f"Model trained on student placement dataset! Accuracy : {accuracy}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    train(n_estimators=100, random_state=42)
    train(n_estimators=200, random_state=42)
    train(n_estimators=150, random_state=0)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Student Placement Prediction")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )[0]

    print(f"\nBest run : {best_run.info.run_name} | Accuracy : {best_run.data.metrics["accuracy"]:.4f}")
    
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    joblib.dump(best_model, models_dir / "model.pkl")

    print(f"Best model saved to models/model.pkl")