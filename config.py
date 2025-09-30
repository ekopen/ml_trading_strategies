# config.py
# variables that are used across the project
import os
# from dotenv import load_dotenv
# load_dotenv() # for when running locally

MARKET_DATA_CLICKHOUSE_IP = "159.203.124.175"
KAFKA_IP = "159.65.41.22:9092"

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")