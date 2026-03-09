import os
import sys
import logging
# os → work with folders/files.
# sys → access system-level streams like stdout.
# logging → Python’s built-in logging system.

logging_dir = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
# Defines the logging format (how every log message will look).
# %(asctime)s → timestamp
# %(levelname)s → log level (INFO, ERROR, WARNING, etc.)
# %(module)s → name of the Python file (module)
# %(message)s → the log message you log
# Final output format example:
# [2025-11-30 10:22:40,123:INFO:train:Training started]

log_dir = "logs"
# Name of folder where logs will be stored.
# Here the folder is "logs".

log_filepath = os.path.join(log_dir, "running_logs.log")
# Creates full path to the log file.
# Example: logs/running_logs.log

os.makedirs(log_dir, exist_ok=True)
# Creates the logs folder if it doesn't already exist.
# exist_ok=True prevents errors if the folder is already present.

logging.basicConfig(level=logging.INFO,
                    format=logging_dir,
                    handlers=[
                        logging.FileHandler(log_filepath),
                        logging.StreamHandler(sys.stdout)
                    ])
# level=logging.INFO
# Only logs with level INFO or higher will be recorded.
# format=logging_dir
# Uses your custom log format defined earlier.
# handlers=[ ... ]
# Logging outputs will go to two places:
# FileHandler(log_filepath)
# Writes logs into logs/running_logs.log.
# StreamHandler(sys.stdout)
# Prints logs on your terminal/console.

logger = logging.getLogger("cnnClassifierLogger")
# Creates a named logger instance called "cnnClassifierLogger".