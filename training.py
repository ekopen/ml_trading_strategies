# training.py
# trains all ML models and publishes them to S3 for the trading model

from config import AWS_BUCKET
from setup import s3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import logging, joblib, os
logger = logging.getLogger(__name__)

def load_dataset(path):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ["price", "future_return", "label","ts"]]
    X = df[feature_cols]
    y = df["label"]
    return X, y

def train_and_eval(X, y, model, name, description, client, s3_key):
    logger.info(f"Training model: {name}")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred, digits=3, output_dict=True, zero_division=0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        # stores model performance and training history in a clickhouse table
        client.insert('model_runs',
            [(name, description, s3_key, train_acc, test_acc, precision, recall, f1)],
            column_names=['model_name', 'model_description', 's3_key', 'train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1'])     
    except Exception as e:
        print(f"Training/evaluation failed for {name}: {e}")
    
def upload_to_cloud(local_path, s3_key):
    try:
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        logger.info(f"Upload complete for {local_path} at s3://{AWS_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"S3 upload failed for {s3_key}: {e}")
        return None 

# trains all models, saves them locally, and uploads to cloud
def train_models(dataset, model_dir, client, models):
    X, y = load_dataset(dataset)
    for model, name, description in models:
        safe_name = name.replace(" ", "_").lower()
        s3_key = f"models/{safe_name}.pkl"
        filename = f"{safe_name}.pkl"
        filepath = f"{model_dir}/{filename}"
        train_and_eval(X, y, model, safe_name, description, client, s3_key)
        joblib.dump(model, filepath)
        upload_to_cloud(filepath, s3_key)



