# feature.py
# creates updated feature tables for both a one minute view and one hour view of the tick data

import logging
logger = logging.getLogger(__name__)

def get_ochlv_minute(client):
    ochlv_df = client.query_df("""
        SELECT avg(price) AS price
        FROM ticks_db
        WHERE timestamp > now() - INTERVAL 4 HOUR
        GROUP BY toStartOfMinute(timestamp)
        ORDER BY toStartOfMinute(timestamp) DESC
        LIMIT 240                                
    """)
    return ochlv_df

def get_ochlv_hour(client):
    ochlv_df = client.query_df("""
        SELECT avg(price) AS price
        FROM ticks_db
        WHERE timestamp > now() - INTERVAL 168 HOUR
        GROUP BY toStartOfHour(timestamp)
        ORDER BY toStartOfHour(timestamp) DESC
        LIMIT 168                                
    """)
    return ochlv_df

def build_features(df):
    # standard features
    df["returns"] = df["price"].pct_change()
    df["sma_20"] = df["price"].rolling(window=20).mean()
    df["sma_60"] = df["price"].rolling(window=60).mean()
    df["momentum_10"] = df["price"] / df["price"].shift(10) - 1
    df["momentum_30"] = df["price"] / df["price"].shift(30) - 1
    df["volatility_30"] = df["returns"].rolling(window=30).std()
    for lag in [1, 2, 5]:
        df[f"lag_return_{lag}"] = df["returns"].shift(lag)

    # RSI
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    df = df.dropna()
    return df

def create_labels(df, horizon, threshold):
    df["future_return"] = df["price"].shift(-horizon) / df["price"] - 1
    df["label"] = 1  # HOLD
    df.loc[df["future_return"] > threshold, "label"] = 2  # BUY
    df.loc[df["future_return"] < -threshold, "label"] = 0  # SHORT/SELL
    df = df.dropna()
    return df

def create_feature_data(client,minute_dir,hour_dir):
    try:
        minute_df = get_ochlv_minute(client)
        minute_df_features = build_features(minute_df)
        minute_df_labels = create_labels(df = minute_df_features, horizon=5, threshold=.002)
        minute_df_labels.to_parquet(minute_dir, index=False)
    except Exception as e:
        logger.exception("Minute feature pipeline failed: {e}")

    try:
        hour_df = get_ochlv_hour(client)
        hour_df_features = build_features(hour_df)
        hour_df_labels = create_labels(df = hour_df_features, horizon=3, threshold=.01)
        hour_df_labels.to_parquet(hour_dir, index=False)
    except Exception as e:
        logger.exception("Hour feature pipeline failed: {e}")