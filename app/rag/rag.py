import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import cohere
# Load environment variables
load_dotenv()


class RAGPipeline:
    def __init__(self, db_handler, embedding_handler):
        """
        Initialize the RAG pipeline with database, embedding handlers, and Cohere client.

        Args:
            db_handler (DatabaseHandler): Instance for database interactions.
            embedding_handler (EmbeddingHandler): Instance for embedding generation.
            cohere_api_key (str): API key for Cohere's services.
        """
        self.db_handler = db_handler
        self.embedding_handler = embedding_handler
        self.cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self.bm25 = None
        self.tokenized_documents = []

    def fetch_data(self, document_type, chunking_type, agent_id):
        """
        Fetch data from the database based on the document type.

        Args:
            document_type (str): The type of document to retrieve ('documents' or 'qa_pairs').
            chunking_type (str): The type of chunking applied.

        Returns:
            list: The retrieved data from the database.
        """
        if document_type == 'documents':
            conditions = f"agent_id = '{agent_id}'"            
            return self.db_handler.fetch_data('agent_upload_docs', columns=['content', 'embedding', 'chunking_type'], conditions=conditions)
        elif document_type == 'qa_pairs':
            return self.db_handler.fetch_data('qa_pairs', columns=['question', 'answer', 'question_embedding'])
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    def calculate_similarities(self, input_embedding, stored_data, is_qa_pairs, top_k=6):
        """
        Calculate cosine similarities between input embedding and stored embeddings.

        Args:
            input_embedding (array): The embedding of the input query.
            stored_data (list): Data fetched from the database.
            is_qa_pairs (bool): Whether the data is from QA pairs or document chunks.
            top_k (int): Number of top results to return.

        Returns:
            list: Top-k similarities sorted in descending order.
        """
        similarities = []

        for record in stored_data:
            try:
                if is_qa_pairs:
                    stored_question = record['question']
                    stored_answer = record['answer']
                    stored_embedding = np.array(
                        eval(record['question_embedding']))
                else:
                    stored_question = None
                    stored_answer = record['content']
                    stored_embedding = np.array(eval(record['embedding']))

                similarity = cosine_similarity(
                    np.array(input_embedding).reshape(1, -1),
                    stored_embedding.reshape(1, -1)
                )[0][0]

                similarities.append({
                    'question': stored_question,
                    'answer': stored_answer,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"Error processing record: {record}, Error: {str(e)}")

        # Sort by similarity and return only top_k results
        sorted_results = sorted(
            similarities, key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_k]

    def rerank_results(self, query, combined_results, top_k):
        """
        Re-rank the combined results using the Cohere re-ranking API.

        Args:
            query (str): The input query.
            combined_results (list): Combined results from similarity and keyword searches.
            top_k (int): Number of top results to return.

        Returns:
            list: Re-ranked results with their original metadata and updated similarity scores.
        """
        # Remove duplicate documents before re-ranking
        seen_texts = set()
        unique_results = []
        for result in combined_results:
            if result['answer'] not in seen_texts:
                unique_results.append(result)
                seen_texts.add(result['answer'])

        # Prepare data for Cohere re-ranking
        documents = [result['answer'] for result in unique_results]

        # Call Cohere re-rank API
        rerank_results = self.cohere_client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model="rerank-english-v2.0"
        )

        # Map re-ranked results back to original data and update similarity scores
        reranked_combined_results = [
            {
                'question': unique_results[doc.index]['question'],
                'answer': unique_results[doc.index]['answer'],
                'similarity': doc.relevance_score,  # Use relevance_score as the similarity score
                'source': unique_results[doc.index]['source'],
                'rank': idx + 1  # Rank based on the re-rank order
            }
            for idx, doc in enumerate(rerank_results.results)
        ]
        return reranked_combined_results[:top_k]


    def retrieve(self, query, agent_id, document_type='documents', top_k=3, chunking_type='agentic'):
        """
        Retrieve the most relevant results based on the specified method.

        Args:
            query (str): The input query.
            document_type (str): The type of document to search in ('documents' or 'qa_pairs').
            top_k (int): The number of top results to return.
            chunking_type (str): The type of chunking applied.
            method (str): The retrieval method ('similarity', 'keyword', 'hybrid').

        Returns:
            list: The top results based on the specified method.
        """
        # Generate embedding for the query (if similarity or hybrid search)
        input_embedding = self.embedding_handler.get_embedding(query)

        # Fetch data based on document type
        stored_data = self.fetch_data(document_type, chunking_type, agent_id)
        is_qa_pairs = (document_type == 'qa_pairs')

        return self.calculate_similarities(input_embedding, stored_data, is_qa_pairs, top_k)
