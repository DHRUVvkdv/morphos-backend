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

            # Create indexes if they don't exist
            db.users.create_index("email", unique=True)
            db.users.create_index("auth0_id", unique=True)

            return db
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")

    logger.warning("MongoDB URI not provided, database features disabled")
    return None


def get_db():
    """Get database connection"""
    global db
    if not db:
        return init_db()
    return db
