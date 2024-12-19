# document_processor.py
from .table_manager import TableManager
from .pdf_processor import PDFProcessor
from app.data.models.models import MetaData

from tiktoken import get_encoding
from data.handlers.embedding_handler import EmbeddingHandler
from data.handlers.db_handler import DatabaseHandler

from openai import OpenAI


class DocumentProcessor:
    def __init__(self, db_config, embedding_config):
        self.db_handler = DatabaseHandler(**db_config)
        self.embedding_handler = EmbeddingHandler(**embedding_config)
        self.encoding = get_encoding('cl100k_base')
        self.client = OpenAI(api_key=embedding_config.get("openai_api_key"))

        self.table_manager = TableManager(self.db_handler, self.client)
        self.pdf_processor = PDFProcessor(
            self.table_manager, self.embedding_handler, self.db_handler, self.encoding)


    def _extract_metadata_values(self, metadata):
        return {
            key: (value.category if isinstance(value, MetaData) else value)
            for key, value in metadata.items()
        }

    def close(self):
        self.db_handler.close_connection()
