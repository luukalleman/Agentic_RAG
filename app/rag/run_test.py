import os
from app.data.handlers.db_handler import DatabaseHandler
from app.data.handlers.embedding_handler import EmbeddingHandler
from dotenv import load_dotenv
from app.rag.rag import RAGPipeline
from app.config.config import get_db_config
from openai import OpenAI
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Initialize OpenAI and environment
client = OpenAI()
source = "documents"

# Queries and corresponding answers
qa_pairs = [
    {"query": "What was Amazon's total revenue in 2023?", "answer": "Amazon's total revenue in 2023 was $574.8 billion."},
    {"query": "Where does Amazon generally invest its excess cash in?", "answer": "Amazon generally invests its excess cash in AAA-rated money market funds and investment grade short- to intermediate-term marketable debt securities"},
    {"query": "What was the year-over-year growth rate of AWS in 2023?", "answer": "The year-over-year growth rate of AWS in 2023 was 12%."},
    {"query": "How much revenue did Amazon's North America segment generate in 2023?", "answer": "Amazon's North America segment generated $353 billion in revenue in 2023."},
    {"query": "What was Amazon's operating income in 2023, and what was the operating margin?", "answer": "Amazon's operating income in 2023 was $36.9 billion with an operating margin of 6.4%."},
    {"query": "What was Amazon's trailing Free Cash Flow for the year 2023?", "answer": "Amazon's Free Cash Flow for 2023 was $35.5 billion."},
    {"query": "How much did Amazon's advertising revenue grow in 2023?", "answer": "Amazon's advertising revenue grew by 24% to $47B in 2023."},
    {"query": "What are the guiding principles or core values that Amazon operates by?", "answer": "Amazon operates by 4 guiding principles: customer obsession rather than competitor focus, passion for invention, commitment to operational excellence, and long-term thinking"},
    {"query": "What are Amazon's key products and services?", "answer": "Amazon's key products and services include e-commerce, AWS cloud services, Prime membership, and Alexa devices."},
    {"query": "How many employees did Amazon report as of December 31, 2023?", "answer": "Amazon reported employed approximately 1,525,000 full-time and part-time employees"},
]

# Define chunking types and retrieval methods
chunking_types = ["agentic", "static", "overlap"]
retrieval_methods = ["hybrid", "similarity", "keyword"]

# Initialize Database and Embedding handlers
db_handler = DatabaseHandler(**get_db_config())
embedding_handler = EmbeddingHandler(
    model_name="openai",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(db_handler, embedding_handler)


def calculate_similarity_scores(query, answer, chunking_type, retrieval_method):
    """
    Calculate re-ranked similarity scores based on the correct answer.
    """
    try:
        # Retrieve top-k results
        results = rag_pipeline.retrieve(
            query,
            document_type=source,
            top_k=6,  # Top results to retrieve
            chunking_type=chunking_type,
            method=retrieval_method
        )

        # Ensure results have all necessary fields
        for result in results:
            result.setdefault("source", retrieval_method)
            result.setdefault("similarity", 0)

        # Re-rank results based on the answer, not the query
        reranked_results = rag_pipeline.rerank_results(answer, results, top_k=3)
        print(f"Chunking: {chunking_type}, Retrieval: {retrieval_method}\n", reranked_results)

        # Sum the similarity scores of the re-ranked results
        total_similarity = sum(result.get("similarity", 0) for result in reranked_results)
        return total_similarity

    except Exception as e:
        print(f"Error during retrieval for {chunking_type} and {retrieval_method}: {e}")
        return 0


def run():
    """
    Main function to iterate over all chunking types and retrieval methods,
    evaluate performance, and display results.
    """
    results_summary = []

    # Iterate over chunking types and retrieval methods
    for chunking_type in chunking_types:
        for retrieval_method in retrieval_methods:
            total_similarity_sum = 0
            for qa_pair in qa_pairs:
                query = qa_pair["query"]
                answer = qa_pair["answer"]
                total_similarity_sum += calculate_similarity_scores(query, answer, chunking_type, retrieval_method)

            # Append results
            results_summary.append({
                "Chunking Type": chunking_type,
                "Retrieval Method": retrieval_method,
                "Total Similarity Score": total_similarity_sum
            })

    # Sort results by Total Similarity Score in descending order
    results_summary = sorted(results_summary, key=lambda x: x["Total Similarity Score"], reverse=True)

    # Display the results
    print(tabulate(results_summary, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    run()