from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


default_args = {"owner": "airflow", "retries": 1}


with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Train → Evaluate → Promote model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # ------------------------
    # 1. TRAIN MODEL
    # ------------------------
    def train_model_task():
        data = load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)

        # Save model + test data for evaluation
        joblib.dump((model, X_test, y_test), "models/model.pkl")


    # ------------------------
    # 2. EVALUATE MODEL
    # ------------------------
    def evaluate_model_task():
        model, X_test, y_test = joblib.load("models/model.pkl")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        metrics = {"accuracy": acc}

        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"Model accuracy: {acc}")


    # ------------------------
    # 3. PROMOTE MODEL
    def promote_model_task():
        import boto3
        
        # Ensure models folder exists
        os.makedirs("models", exist_ok=True)


        with open("models/metrics.json") as f:
            metrics = json.load(f)

        acc = metrics["accuracy"]

        # QUALITY GATE
        if acc < 0.94:
            raise ValueError(f"Model failed threshold: {acc}")

        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        metadata = {
          "model_version": version,
          "dataset": "breast_cancer",
          "model_type": "logistic_regression",
          "accuracy": acc,
        }

        # Save metadata locally
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Model promoted locally! Version: {version}")

        # ------------------------
        # S3 UPLOAD
        # ------------------------
        s3 = boto3.client("s3")

        bucket_name = "mlops-models-s3-bucket"   #  my bucket name
        prefix = f"models/{version}/"

        s3.upload_file("models/model.pkl", bucket_name, prefix + "model.pkl")
        s3.upload_file("models/metrics.json", bucket_name, prefix + "metrics.json")
        s3.upload_file("models/metadata.json", bucket_name, prefix + "metadata.json")

        print(f"Uploaded model artifacts to S3 at {prefix}")
    
    
    # ------------------------
    # TASKS
    # ------------------------
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )

    eval_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_task,
    )

    train_task >> eval_task >> promote_task