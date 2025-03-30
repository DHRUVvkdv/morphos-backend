# config/database.py
import os
import logging

logger = logging.getLogger("morphos-db-config")


class MongoDBSettings:
    """MongoDB settings without Pydantic"""

    def __init__(self):
        self.MONGODB_URI = os.environ.get("MONGODB_URI", "")
        self.MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "morphos_db")


# Instantiate settings
mongodb_settings = MongoDBSettings()

# Log MongoDB settings
if mongodb_settings.MONGODB_URI:
    logger.info("MongoDB settings loaded successfully")
    logger.info(f"MONGODB_DB_NAME: {mongodb_settings.MONGODB_DB_NAME}")
else:
    logger.warning("MongoDB URI not provided, database features will be limited")
