import os 
import logging
from datetime import datetime

log_dir = r"E:\new_mlops\logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_file_path = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

'''if __name__ == "__main__":
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")'''
