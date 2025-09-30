# main.py
# starts running the ML pipeline

import threading, signal, logging, schedule, time
from setup import market_clickhouse_client, ml_clickhouse_client
from features import create_feature_data
from training import train_models
from models import models
from logging.handlers import RotatingFileHandler

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

# directories and data
feature_dir = "feature_data/feature_data.parquet"
model_dir = "models"
market_client = market_clickhouse_client()
ml_client = ml_clickhouse_client()

# run end to end pipeline
def run_model_pipeline():
    create_feature_data(market_client, feature_dir)
    train_models(feature_dir, model_dir, ml_client, models)

# start/stop loop
if __name__ == "__main__":
    try:
        logger.info("System starting.")

        # run once immediately
        run_model_pipeline()

        # these currently run every 12 hours from the START of the program, vs the clock option
        # schedule.every().day.at("00:00").do(run_model_pipeline)
        schedule.every(12).hours.do(run_model_pipeline)

        while not stop_event.is_set():
             schedule.run_pending()
             time.sleep(1)

        logger.info("System shutdown complete.") 

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
    except Exception:
        logger.exception("Fatal error in main loop")
        