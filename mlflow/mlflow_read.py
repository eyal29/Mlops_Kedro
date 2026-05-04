from mlflow.tracking import MlflowClient

# connexion à MLflow
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# 👉 remplace par ton vrai run_id
run_id = "2e40c1edc09340a4a95883cb4bf01ded"

# récupérer l’historique de la métrique f1
metrics = client.get_metric_history(run_id, key="f1")

print(metrics)
