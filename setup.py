# setup.py
# one off container to run at the start
# docker compose run --rm db_setup

from config import MARKET_DATA_CLICKHOUSE_IP, AWS_ACCESS_KEY, AWS_SECRET_KEY
import clickhouse_connect
import logging, boto3
logger = logging.getLogger(__name__)

def market_clickhouse_client():      
    client = clickhouse_connect.get_client(
        host=MARKET_DATA_CLICKHOUSE_IP,
        port=8123,
        username="default",   
        password="mysecurepassword",
        database="default"
    )
    return client

def ml_clickhouse_client():
    return clickhouse_connect.get_client(
        host="localhost", #clickhouse for docker, localhost for local dev 
        port=8123,
        username="default",
        password="mysecurepassword",
        database="default"
    )

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# try:
#     logger.info("Deleting tables.")
#     ch = ml_clickhouse_client()
#     ch.command("DROP TABLE IF EXISTS model_runs SYNC")
# except Exception:
#     logger.warning("Error when deleting tables.")

try:
    logger.info("Creating tables.")
    ch = ml_clickhouse_client()
    ch.command('''
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id UUID DEFAULT generateUUIDv4(),
        trained_at DateTime DEFAULT now(),
        model_name String,
        s3_key String,
        train_accuracy Float64,
        test_accuracy Float64,
        precision Float64,
        recall Float64,
        f1 Float64               
    ) 
    ENGINE = MergeTree()
    ORDER BY (trained_at);
    ''')
except Exception:
    logger.warning("Error when creating tables.")

