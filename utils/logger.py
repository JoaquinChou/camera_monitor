from loguru import logger
import sys



def setup_logger(std_level, file_level, log_file):
    logger.remove()

    logger.add(sys.stdout,
               format="{time:YYYY-MM-DD HH:mm:ss:SSS} | {level} | {file}:{line} - {message}",
               level=std_level)
    logger.add(log_file,
               rotation="1 week",
               retention="1 month",
               level=file_level,
               format="{time:YYYY-MM-DD HH:mm:ss:SSS} | {level} | {file}:{line} - {message}")
    return logger


logger = setup_logger('INFO', 'INFO', log_file='logs/camera_monitor.log')