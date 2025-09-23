from config import market_data_clickhouse_ip
import numpy as np
import pandas as pd
import clickhouse_connect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
logger = logging.getLogger(__name__)

def market_clickhouse_client():      
    client = clickhouse_connect.get_client(
        host=market_data_clickhouse_ip,
        port=8123,
        username="default",   
        password="mysecurepassword",
        database="default"
    )
    return client


def get_ochlv(client):
    ochlv_df = client.query_df("""
        SELECT avg(price) AS price
        FROM ticks_db
        WHERE timestamp > now() - INTERVAL 2 HOUR
        GROUP BY toStartOfMinute(timestamp)
        ORDER BY toStartOfMinute(timestamp) DESC
        LIMIT 120                                
    """)
    return ochlv_df

def build_features(df):
    df["returns"] = df["price"].pct_change()
    df["sma_20"] = df["price"].rolling(20).mean()
    df["sma_60"] = df["price"].rolling(60).mean()
    df["volatility"] = df["returns"].rolling(30).std()
    df = df.dropna()
    return df

def create_labels(df, horizon):
    df["future_return"] = df["price"].shift(-horizon) / df["price"] - 1
    df["label"] = 1  # default hold value
    df.loc[df["future_return"] > 0.002, "label"] = 2  # BUY
    df.loc[df["future_return"] < -0.002, "label"] = 0  # SHORT/SELL
    df = df.dropna()
    return df

market_data_client = market_clickhouse_client()
ochlv_df = get_ochlv(market_data_client)
feature_df = build_features(ochlv_df)
labels = create_labels(feature_df, 5)

X = feature_df[["returns", "sma_20", "sma_60", "volatility"]]
y = labels["label"]
X, y = X.loc[y.index], y #aligning indexes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# check accuracy
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(clf, model_path)

print(f"Model saved to {model_path}")
