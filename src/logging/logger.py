import os
import logging.config

def setup_logger(log_dir=None, train_mode=True):
    """
    Set up the logger with optional log directory.

    Parameters:
    - log_dir: Directory to save the log file. If None, logs are only written to the console.

    Returns:
    - logger: Configured logger instance.
    """
    # Default logging configuration
    DEFAULT_LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Logs to the console
            },
        },
        'loggers': {
            'log': {
                'handlers': ['default'],  # Default handler (console)
                'level': 'INFO',
                'propagate': True
            }
        }
    }

    # Add a FileHandler if log_dir is provided
    if log_dir is not None:
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        if train_mode:
            # Define the log file path
            log_file = os.path.join(log_dir, 'training.log')
        else:
            # Define the log file path
            log_file = os.path.join(log_dir, 'prediction.log')

        # Add the FileHandler configuration
        DEFAULT_LOGGING['handlers']['file_handler'] = {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': log_file,
            'mode': 'a',  # Append mode ('w' for overwrite)
        }

        # Update the logger to include the file handler
        DEFAULT_LOGGING['loggers']['log']['handlers'].append('file_handler')

    # Apply the logging configuration
    logging.config.dictConfig(DEFAULT_LOGGING)

    # Create and return the logger
    return logging.getLogger('log')


# Example Usage
if __name__ == "__main__":
    # Specify the log directory (optional)
    log_directory = "logs"  # Change this to your desired directory or set to None

    # Set up the logger
    logger = setup_logger(log_dir=log_directory)

    # Example log messages
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")