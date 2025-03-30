from pymongo import MongoClient
import logging
import os
from config.database import mongodb_settings

logger = logging.getLogger("morphos-db")

# Initialize MongoDB connection
client = None
db = None


def init_db():
    """Initialize database connection"""
    global client, db

    # First try to use mongodb_settings
    mongodb_uri = None
    db_name = "morphos_db"

    if mongodb_settings:
        mongodb_uri = mongodb_settings.MONGODB_URI
        db_name = mongodb_settings.MONGODB_DB_NAME
    else:
        # Fallback to direct environment variable
        mongodb_uri = os.environ.get("MONGODB_URI")

    if mongodb_uri:
        try:
            client = MongoClient(mongodb_uri)
            db = client[db_name]
            logger.info(f"Connected to MongoDB: {db_name}")

            # Don't perform boolean test on db object
            return db
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            return None

    logger.warning("MongoDB URI not provided, database features disabled")
    return None


def get_db():
    """Get database connection"""
    global db
    if db is None:
        return init_db()
    return db
