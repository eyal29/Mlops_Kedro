import mlflow

mlflow.set_tracking_uri("http://34.133.16.86")

client = mlflow.MlflowClient()
experiments = client.search_experiments()

print("Connexion OK")
print([exp.name for exp in experiments])
