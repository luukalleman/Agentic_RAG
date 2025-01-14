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
    import json

    def process_pdf(self, file_path, document_title, document_metadata, agent_id, chunk_type="static"):
        """
        Reads a PDF file, extracts its content, and processes it using the specified chunker type.
        """
        document_content = self._extract_text_from_pdf(file_path)

        self.table_manager.create_table(
            table_name="Agent_Upload_Docs",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "title": "TEXT",
                "content": "TEXT",  # Store plain text content
                "embedding": "VECTOR(1536)",
                "metadata": "JSONB",  # Metadata should be JSON
                "chunking_type": "TEXT",
                "agent_id": "INTEGER"
            },
            raw_data=document_content[:1000]
        )

        # Initialize the chunker based on the specified type
        chunker = ChunkerFactory.create_chunker(chunk_type, document_content)

        # Use the chunker to process the document
        structured_results = chunker.process_document()

        for chunk_group in structured_results:
            chunk_text = " ".join(chunk_group.sentences)  # Join the list of sentences into a single string of text

            # Instead of trying to adapt a dict, ensure only text is written to the content column
            if not isinstance(chunk_text, str):
                chunk_text = str(chunk_text)  # Convert to string if needed

            embedding = self.embedding_handler.get_embedding(chunk_text)

            data = {
                "title": document_title,
                "content": chunk_text,  # This should only be plain text now
                "embedding": embedding,
                "metadata": json.dumps(document_metadata),  # Convert metadata dict to JSON string
                "chunking_type": chunk_type,
                "agent_id": agent_id
            }

            try:
                self.db_handler.insert_row('Agent_Upload_Docs', data)
            except Exception as e:
                print(f"Error inserting row: {str(e)}")
            
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
