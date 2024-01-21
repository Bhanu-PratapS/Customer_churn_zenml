import os 
import logging
from datetime import datetime

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)  # Creates the directory if it doesn't exist

    log_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_file_path = os.path.join(log_dir, log_file)

    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    return logging.getLogger(__name__)

logger = setup_logger(r"E:\new_mlops\logs")
