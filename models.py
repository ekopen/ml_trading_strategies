# models.py
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET
import pandas as pd
import joblib, os, boto3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

def load_dataset(path):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ["price", "future_return", "label"]]
    X = df[feature_cols]
    y = df["label"]
    return X, y

def train_and_eval(X, y, model, name, ch, s3_key):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    ch.insert('model_runs',
        [(name, s3_key, train_acc, test_acc, precision, recall, f1)],
        column_names=['model_name', 's3_key', 'train_accuracy', 'test_accuracy', 'precision', 'recall', 'f1'])
    
def upload_to_cloud(local_path, s3_key):
    s3.upload_file(local_path, AWS_BUCKET, s3_key)
    return f"s3://{AWS_BUCKET}/{s3_key}"
        
def generate_models(dataset, model_dir):
    X, y = load_dataset(dataset)
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    models = [
        (LogisticRegression(max_iter=500), f"Logistic Regression - {dataset_name}"),
        (RandomForestClassifier(n_estimators=100, random_state=42), f"Random Forest - {dataset_name}"),
        (GradientBoostingClassifier(n_estimators=100, random_state=42), f"Gradient Boosting - {dataset_name}")
    ]
    for model, name in models:
        safe_name = name.replace(" ", "_").lower()
        s3_key = f"models/{safe_name}.pkl"
        filename = f"{safe_name}.pkl"
        filepath = f"{model_dir}/{filename}"
        train_and_eval(X, y, model, safe_name, s3_key)
        joblib.dump(model, filepath)
        upload_to_cloud(filepath, s3_key)



