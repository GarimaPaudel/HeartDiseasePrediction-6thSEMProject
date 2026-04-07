import os
from loguru import logger

LOG_DIR = "artifacts/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default stderr sink and add custom ones
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.add(
    sink=os.path.join(LOG_DIR, "pipeline_{time:YYYYMMDD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)
