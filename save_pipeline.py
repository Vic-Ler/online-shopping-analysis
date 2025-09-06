import mlflow.sklearn
import joblib
import os
import mlflow

zenml_mlflow_store = r"C:\Users\Lenovo\AppData\Roaming\zenml\local_stores\0e9e92e9-2d3d-45c5-8fc4-da7417f62fbe\mlruns"
mlflow.set_tracking_uri(f"file:{zenml_mlflow_store}")

mlflow_model_uri = "runs:/e0e7b9f978e44ab9a513d526d254ce87/model"
pipeline = mlflow.sklearn.load_model(mlflow_model_uri)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")

print("Pipeline saved as models/pipeline.pkl")

#run only once to save pipeline with id found in mlflow