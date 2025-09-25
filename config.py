# config.py
# variables that are used across the project
import os

MARKET_DATA_CLICKHOUSE_IP = "159.203.124.175"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
