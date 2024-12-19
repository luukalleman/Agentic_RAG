# pdf_processor.py
import json
from app.rag.chunking import ChunkerFactory
import PyPDF2


class PDFProcessor:
    def __init__(self, table_manager, embedding_handler, db_handler, encoding):
        self.table_manager = table_manager
        self.embedding_handler = embedding_handler
        self.db_handler = db_handler
        self.encoding = encoding

    def process_pdf(self, file_path, document_title, document_metadata, chunk_type="agentic"):
        """
        Reads a PDF file, extracts its content, and processes it using the specified chunker type.
        """
        document_content = self._extract_text_from_pdf(file_path)

        self.table_manager.create_table(
            table_name="documents",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "title": "TEXT",
                "content": "TEXT",
                "embedding": "VECTOR(1536)",
                "metadata": "JSONB",
                "chunking_type": "TEXT"
            },
            raw_data=document_content[:1000]
        )

        # Initialize the chunker based on the specified type
        chunker = ChunkerFactory.create_chunker(
            chunk_type, document_content)

        # Use the chunker to process the document
        structured_results = chunker.process_document()

        # Process each chunk group and insert into the database
        for chunk_group in structured_results:
            chunk_text = chunk_group.sentences  # Use the rewritten text directly

            embedding = self.embedding_handler.get_embedding(chunk_text)
            data = {
                "title": document_title,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": json.dumps(document_metadata),
                "chunking_type": chunk_type  # Add the chunking type to the row
            }
            self.db_handler.insert_row('documents', data)
            self.close()
            
    def _extract_text_from_pdf(self, file_path):
        """
        Extracts text from a PDF file.
        """
        text = ""
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
