import logging
import mlflow
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_latest_pipeline() -> object:
    """
    Load the latest trained pipeline from ZenML's MLflow store.
    """
    zenml_mlflow_store = (
        r"C:\Users\Lenovo\AppData\Roaming\zenml\local_stores"
        r"\0e9e92e9-2d3d-45c5-8fc4-da7417f62fbe\mlruns"
    )
    mlflow.set_tracking_uri(f"file:{zenml_mlflow_store}")
    logging.info(f"MLflow tracking URI set to ZenML store: {zenml_mlflow_store}")

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        raise ValueError("No experiments found in the ZenML MLflow store!")

    latest_experiment = max(experiments, key=lambda e: e.creation_time)
    experiment_id = latest_experiment.experiment_id
    logging.info(
        f"Using latest experiment: {latest_experiment.name} (ID: {experiment_id})"
    )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{latest_experiment.name}'.")

    latest_run_id = runs[0].info.run_id
    logging.info(f"Latest ZenML MLflow run ID: {latest_run_id}")

    mlflow_model_uri = f"runs:/{latest_run_id}/model"
    logging.info(f"Loading pipeline from MLflow: {mlflow_model_uri}")
    pipeline = mlflow.sklearn.load_model(mlflow_model_uri)

    logging.info("Pipeline loaded successfully.")
    return pipeline