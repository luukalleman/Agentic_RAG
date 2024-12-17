import os
import json
import openai
import tiktoken
from sentence_transformers import SentenceTransformer

def chunk_text(text, max_tokens, encoding_name='cl100k_base'):
    """
    Splits text into chunks that fit within the specified token limit.
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): Maximum number of tokens per chunk.
        encoding_name (str): The encoding name used for tokenization.
    Returns:
        list: A list of text chunks.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = len(encoding.encode(word))
        if current_tokens + word_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

class EmbeddingHandler:
    def __init__(self, model_name='openai', openai_api_key=None):
        """
        Initialize the embedding handler with the specified model.
        Args:
            model_name (str): The name of the embedding model to use ('openai' or any Sentence-Transformer model).
            openai_api_key (str, optional): OpenAI API key, required if using OpenAI model.
        """
        self.model_name = model_name
        if model_name == 'openai':
            if not openai_api_key:
                raise ValueError("OpenAI API key must be provided for OpenAI embeddings.")
            openai.api_key = openai_api_key
        else:
            self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        """
        Generate embeddings for the given text using the specified model.
        Args:
            text (str): The input text to embed.
        Returns:
            list: The embedding vector.
        """
        if self.model_name == 'openai':
            response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
            return response.data[0].embedding
        else:
            return self.model.encode(text).tolist()

def process_document(db_handler, embedding_handler, title, content, metadata=None):
    """
    Process a document by chunking its content, generating embeddings, and storing them in the database.
    Args:
        db_handler (DatabaseHandler): The database handler instance.
        embedding_handler (EmbeddingHandler): The embedding handler instance.
        title (str): The title of the document.
        content (str): The content of the document.
        metadata (dict, optional): Additional metadata for the document.
    """
    max_tokens = 8191  # Maximum tokens for 'text-embedding-3-small'
    chunks = chunk_text(content, max_tokens)
    for chunk in chunks:
        embedding = embedding_handler.get_embedding(chunk)
        data = {
            'title': title,
            'content': chunk,
            'embedding': embedding,
            'metadata': json.dumps(metadata) if metadata else None
        }
        db_handler.insert_row('documents', data)

def process_qa_pair(db_handler, embedding_handler, question, answer, metadata=None):
    """
    Process a Q&A pair by generating embeddings for the question and answer, and storing them in the database.
    Args:
        db_handler (DatabaseHandler): The database handler instance.
        embedding_handler (EmbeddingHandler): The embedding handler instance.
        question (str): The question text.
        answer (str): The answer text.
        metadata (dict, optional): Additional metadata for the Q&A pair.
    """
    question_embedding = embedding_handler.get_embedding(question)
    answer_embedding = embedding_handler.get_embedding(answer)
    data = {
        'question': question,
        'question_embedding': question_embedding,
        'answer': answer,
        'answer_embedding': answer_embedding,
        'metadata': json.dumps(metadata) if metadata else None
    }
    db_handler.insert_row('qa_pairs', data)