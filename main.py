# main.py

import threading, signal, logging, schedule, time
from setup import market_clickhouse_client, ml_clickhouse_client
from features import market_clickhouse_client, create_feature_data
from models import generate_models

# logging 
from logging.handlers import RotatingFileHandler
# logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        RotatingFileHandler(
            "log_data/app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# shutdown
stop_event = threading.Event()
def handle_signal(signum, frame):
    logger.info("Received stop signal. Shutting down...")
    stop_event.set()
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal) # CTRL+C shutdown

minute_data_dir = "feature_data/minute_features.parquet"
hour_data_dir = "feature_data/hour_features.parquet"
model_dir = "models"
market_client = market_clickhouse_client()
ml_client = ml_clickhouse_client()

def job_minute():
    create_feature_data(market_client, minute_data_dir, hour_data_dir)
    generate_models(minute_data_dir, model_dir, ml_client)

def job_hour():
    create_feature_data(market_client, minute_data_dir, hour_data_dir)
    generate_models(hour_data_dir, model_dir, ml_client)

# start/stop loop
if __name__ == "__main__":
    try:
        logger.info("System starting.")

        # these currently run every hour/24 hours from the START of the program, remove the 1 and 24 if we want to follow strict time rules
        schedule.every(1).hour.at(":00").do(job_minute)
        schedule.every(24).day.at("00:00").do(job_hour)

        while not stop_event.is_set():
             schedule.run_pending()
             time.sleep(1)

        logger.info("System shutdown complete.") 

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
    except Exception:
        logger.exception("Fatal error in main loop")
        