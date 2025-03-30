from pymongo import MongoClient
from config.auth import auth0_settings
import logging

logger = logging.getLogger("morphos-db")

# Initialize MongoDB connection
client = None
db = None


def init_db():
    """Initialize database connection"""
    global client, db

    if auth0_settings.MONGODB_URI:
        try:
            client = MongoClient(auth0_settings.MONGODB_URI)
            db = client[auth0_settings.MONGODB_DB_NAME]
            logger.info(f"Connected to MongoDB: {auth0_settings.MONGODB_DB_NAME}")

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
