from app.data.insert.document_processor import DocumentProcessor
from app.config.config import get_db_config, get_embedding_config

# Initialize your processor just like before
processor = DocumentProcessor(get_db_config(), get_embedding_config())

# To process a PDF document
pdf_file_path = "app/data/input/Flowwise_Ecommerce_Documentation.pdf"
processor.pdf_processor.process_pdf(pdf_file_path, "flowise info","Amazon Info", 12345, chunk_type='static')