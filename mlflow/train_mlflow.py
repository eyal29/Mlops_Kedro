import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mlflow
import mlflow.sklearn
from lightgbm.sklearn import LGBMClassifier  # Wrapper pour scikit-learn
from mlflow.models import infer_signature
from sklearn.metrics import (
    f1_score,
    PrecisionRecallDisplay,
    precision_recall_curve,
)
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conf/local/mlflow-client.json"
# Authentification à Google Cloud avec la clé correspondant au compte de service MLflow
# Nouvel URI de l'interface MLflow
client = storage.Client()  # Mettez l'adresse de Google Compute Engine ici


# Mettez le chemin de votre kedro local ici, nous allons lancer les cellules pour la suite
X_train = pd.read_csv(os.path.expanduser("data/05_model_input/X_train.csv"))
X_test = pd.read_csv(os.path.expanduser("data/05_model_input/X_test.csv"))
y_train = pd.read_csv(os.path.expanduser("data/05_model_input/y_train.csv")).values.flatten()
y_test = pd.read_csv(os.path.expanduser("data/05_model_input/y_test.csv")).values.flatten()


int_columns = [
    "product_id",
    "brand",
    "num_views_session",
    "num_views_product",
    "category",
    "sub_category",
    "hour",
    "minute",
    "weekday",
    "duration",
    "num_prev_sessions",
    "num_prev_product_views",
]
X_train[int_columns] = X_train[int_columns].astype("float64")
X_test[int_columns] = X_test[int_columns].astype("float64")
# Hyper-paramètres des modèles
hyp_params = {
    "num_leaves": 60,
    "min_child_samples": 10,
    "max_depth": 12,
    "n_estimators": 100,
    "learning_rate": 0.1,
}

# Identification de l'interface MLflow
mlflow.set_tracking_uri("http://34.133.16.86")
mlflow.set_experiment("purchase_predict")


def save_pr_curve(X, y, model):
    plt.figure(figsize=(16, 11))
    prec, recall, _ = precision_recall_curve(
        y,
        model.predict_proba(X)[:, 1],
        pos_label=1,
    )
    PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
    plt.title("PR Curve", fontsize=16)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig(os.path.expanduser("data/pr_curve.png"))
    plt.close()


# def train_model_local(params):
#     with mlflow.start_run():
#         model = LGBMClassifier(**params, objective="binary", verbose=-1)
#         model.fit(X_train, y_train)

#         score = f1_score(y_test, model.predict(X_test))
#         save_pr_curve(X_test, y_test, model)

#         mlflow.log_params(params)
#         mlflow.log_metric("f1", score)
#         mlflow.log_artifact(
#             os.path.expanduser("data/pr_curve.png"),
#             artifact_path="plots",
#         )

#         signature = infer_signature(X_train, model.predict(X_train))
#         input_example = X_test.iloc[0:1].copy()

#         mlflow.sklearn.log_model(
#             model,
#             "model",
#             signature=signature,
#             input_example=input_example,
#         )

#         print("F1 score :", score)


# train_model_local({**hyp_params, **{"n_estimators": 200, "learning_rate": 0.05}})
# train_model_local({**hyp_params, **{"n_estimators": 500, "learning_rate": 0.025}})
# train_model_local({**hyp_params, **{"n_estimators": 1000, "learning_rate": 0.01}})


def train_model(params):
    with mlflow.start_run():
        model = LGBMClassifier(**params, objective="binary", verbose=-1)
        model.fit(X_train, y_train)
        score = f1_score(y_test, model.predict(X_test))
        save_pr_curve(X_test, y_test, model)
        mlflow.log_params(hyp_params)
        mlflow.log_metric("f1", score)
        mlflow.log_artifact(os.path.expanduser("data/pr_curve.png"), artifact_path="plots")
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_test.iloc[0:1].copy()
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
        )


train_model({**hyp_params, **{"n_estimators": 200, "learning_rate": 0.05}})
