# qa_processor.py
import json
import pandas as pd
from app.data.models.models import MetaData


class QAPairProcessor:
    def __init__(self, table_manager, embedding_handler, db_handler, client):
        self.table_manager = table_manager
        self.embedding_handler = embedding_handler
        self.db_handler = db_handler
        self.client = client

    def process_qa_pair(self, question, answer, metadata=None):
        question_embedding = self.embedding_handler.get_embedding(question)
        answer_embedding = self.embedding_handler.get_embedding(answer)

        # Serialize metadata if present
        metadata_serializable = json.dumps(
            self._serialize_metadata(metadata)) if metadata else None

        data = {
            "question": question,
            "question_embedding": question_embedding,
            "answer": answer,
            "answer_embedding": answer_embedding,
            "metadata": metadata_serializable
        }

        self.db_handler.insert_row('qa_pairs', data)

    def process_qa_from_excel(self, file_path):
        df = pd.read_excel(file_path)
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            raise ValueError(
                "Excel file must contain 'Question' and 'Answer' columns.")

        raw_data = "\n".join(
            f"Q: {row['Question']} | A: {row['Answer']}" for _, row in df.iterrows()
        )

        self.table_manager.create_table(
            table_name="qa_pairs",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "question": "TEXT",
                "answer": "TEXT",
                "question_embedding": "VECTOR(1536)",
                "answer_embedding": "VECTOR(1536)",
                "metadata": "JSONB"
            },
            raw_data=raw_data[:1000]
        )

        for _, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']
            metadata = self._generate_metadata(question, answer)
            self.process_qa_pair(question, answer, metadata)

    def _serialize_metadata(self, metadata):
        if isinstance(metadata, MetaData):
            # Convert `MetaData` to dict
            return {"category": metadata.category}
        elif isinstance(metadata, dict):
            # Recursively serialize dictionaries
            return {key: self._serialize_metadata(value) for key, value in metadata.items()}
        elif isinstance(metadata, list):
            # Recursively serialize lists
            return [self._serialize_metadata(item) for item in metadata]
        return metadata  # Return as-is if already serializable

    def _generate_metadata(self, question, answer):
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Extract the category of the question for this ecom store."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\nProvide a suitable category."}
            ],
            response_format=MetaData,
        )
        # Safely extract and return as a dictionary
        parsed_metadata = completion.choices[0].message.parsed
        if isinstance(parsed_metadata, MetaData):
            # Convert to dictionary
            return {"category": parsed_metadata.category}
        return {"category": "Unknown"}  # Fallback case
