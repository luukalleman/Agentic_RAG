from dotenv import load_dotenv
import os

load_dotenv()

def get_db_config():
# Initialize database handler for dynamic updates
    db_config = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }
    return db_config

def get_embedding_config():
# Initialize database handler for dynamic updates
    embedding_config = {
        "model_name": "openai",
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    return embedding_config