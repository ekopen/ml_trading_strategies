# feature.py
# creates updated feature tables for both a one minute view and one hour view of the tick data

import logging
logger = logging.getLogger(__name__)

def get_ochlv_seconds(client): #should yield about 40k rows to train on
    ochlv_df = client.query_df("""
        SELECT 
            toStartOfInterval(timestamp, INTERVAL 15 SECOND) AS ts, 
            avg(price) AS price
        FROM ticks_db
        GROUP BY ts
        ORDER BY ts DESC
        LIMIT 40320
    """)
    return ochlv_df

def get_ochlv_minute(client): #should yield about 1000 rows to train on
    ochlv_df = client.query_df("""
        SELECT toStartOfMinute(timestamp) AS ts, avg(price) AS price
        FROM ticks_db
        GROUP BY toStartOfMinute(timestamp)
        ORDER BY toStartOfMinute(timestamp) DESC
        LIMIT 1080                                
    """)
    return ochlv_df

def build_features(df):
    # standard features
    df["returns"] = df["price"].pct_change()
    df["sma_30"] = df["price"].rolling(window=30).mean()
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
    df["14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    df = df.dropna()
    return df

def create_labels(df, horizon, threshold):
    df = df.copy()  # make sure it's not a view
    df["future_return"] = df["price"].shift(-horizon) / df["price"] - 1
    df["label"] = 1 # HOLD
    df.loc[df["future_return"] > threshold, "label"] = 2 # BUY
    df.loc[df["future_return"] < -threshold, "label"] = 0 # SELL
    return df.dropna()

def create_feature_data(client,second_dir,minute_dir):
    try:
        second_df = get_ochlv_seconds(client)
        second_df_features = build_features(second_df)
        second_df_labels = create_labels(df = second_df_features, horizon=20, threshold=.0002)
        second_df_labels.to_parquet(second_dir, index=False)
    except Exception as e:
        logger.exception(f"Second feature pipeline failed: {e}")

    try:
        minute_df = get_ochlv_minute(client)
        minute_df_features = build_features(minute_df)
        minute_df_labels = create_labels(df = minute_df_features, horizon=5, threshold=.001)
        minute_df_labels.to_parquet(minute_dir, index=False)
    except Exception as e:
        logger.exception(f"Minute feature pipeline failed: {e}")