import openai
from pydantic import BaseModel
from typing import List
from collections import deque
from app.data.models.models import ChunkGroups, ChunkGroupSchema, ChunkGroupsDirect, ChunkGroupSchemaDirect

class BaseChunker:
    """
    A base class for all chunkers.
    """

    def __init__(self, document_text, max_chunk_size=750, batch_size=50):
        self.document_text = document_text
        self.max_chunk_size = max_chunk_size
        self.batch_size = batch_size
        self.sentences = []
        self.sentence_ids = []

    def split_into_sentences(self):
        """
        Splits the document text into sentences and assigns IDs.
        """
        import re
        sentence_endings = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        self.sentences = [sentence.strip() for sentence in sentence_endings.split(
            self.document_text) if sentence]
        self.sentence_ids = list(range(1, len(self.sentences) + 1))
        return self.sentences

    def process_document(self):
        """
        Processes the document. To be implemented by subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    
class AgenticChunker(BaseChunker):
    """
    A class to handle Agentic Chunking with structured outputs from the LLM.
    """

    def _process_batch(self, sentences):
        combined_sentences = "\n".join(sentences)
        prompt = (
            "You are an expert at rewriting text to improve the quality of retrieving the correct context via our RAG similarity search system."
            "Below is a list of sentences. Your task is to group the sentences into coherent short chunks based on their context and meaning."
            "Once grouped, rewrite the group into one or two short, clear, and descriptive sentences that retain all the information from the original sentences."
            "The rewritten sentences should be self-contained, ensuring they can be understood independently without requiring external context.\n\n"
            "Sentences:\n"
            f"{combined_sentences}\n\n"
            "Return the output as structured JSON with the following format:\n"
            "{\n"
            "  'chunks': [\n"
            "    {\n"
            "      'rewritten': 'The rewritten text that combines and improves the grouped sentences.'\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Each chunk should be represented as a single JSON object with the 'rewritten' field containing the rewritten text."
        )

        try:
            client = openai.OpenAI()
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert text chunker and rewriter."},
                    {"role": "user", "content": prompt},
                ],
                response_format=ChunkGroupsDirect,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"An error occurred while calling the LLM: {e}")
            return None

    def process_with_llm(self):
        all_results = []

        for start in range(0, len(self.sentences), self.batch_size):
            end = start + self.batch_size
            batch_sentences = self.sentences[start:end]
            result = self._process_batch(batch_sentences)
            if result:
                all_results.extend(result.chunks)
            else:
                print(f"No results returned for batch {start + 1}-{end}")

        return all_results

    def process_document(self):
        self.split_into_sentences()
        return self.process_with_llm()
class StaticChunker(BaseChunker):
    """
    A simple static chunker that splits sentences into fixed-size chunks.
    """

    def process_document(self):
        self.split_into_sentences()
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in self.sentences:  # Use the actual sentence text
            if current_size + len(sentence) <= self.max_chunk_size:
                current_chunk.append(sentence)  # Append actual sentence
                current_size += len(sentence)
            else:
                # Append current chunk group with the full sentence text
                chunks.append(
                    ChunkGroupSchema(
                        chunk_id=len(chunks) + 1,
                        sentences=current_chunk,  # Full sentence list
                        reason="Static chunking by size"
                    )
                )
                current_chunk = [sentence]  # Start new chunk with the sentence
                current_size = len(sentence)

        if current_chunk:
            chunks.append(
                ChunkGroupSchema(
                    chunk_id=len(chunks) + 1,
                    sentences=current_chunk,
                    reason="Static chunking by size"
                )
            )
        return chunks
class OverlapChunker(BaseChunker):
    """
    An Overlap chunker for hierarchical chunking logic with overlapping chunks.
    """

    def __init__(self, document_text, max_chunk_size=5, overlap_size=1, batch_size=1):
        super().__init__(document_text, max_chunk_size, batch_size)
        self.overlap_size = overlap_size

    def process_document(self):
        """
        Process the document into overlapping chunks.
        """
        self.split_into_sentences()

        chunks = []
        queue = deque(maxlen=self.max_chunk_size)
        chunk_id = 1

        # Iterator over sentences and their IDs
        sentence_iterator = iter(zip(self.sentence_ids, self.sentences))

        try:
            # Pre-fill the queue with the first chunk
            for _ in range(self.max_chunk_size):
                sentence_id, sentence = next(sentence_iterator)
                queue.append(sentence_id)

            while True:
                # Append current chunk to the result
                chunks.append(
                    ChunkGroupSchema(
                        chunk_id=chunk_id,
                        sentences=list(queue),
                        reason="Overlap chunking",
                    )
                )
                chunk_id += 1

                # Add new sentences to the queue for the next chunk
                for _ in range(self.max_chunk_size - self.overlap_size):
                    sentence_id, sentence = next(sentence_iterator)
                    queue.append(sentence_id)
        except StopIteration:
            # Handle any remaining elements in the queue
            if queue:
                chunks.append(
                    ChunkGroupSchema(
                        chunk_id=chunk_id,
                        sentences=list(queue),
                        reason="Overlap chunking",
                    )
                )

        return chunks


class ChunkerFactory:
    """
    A factory class to create different types of chunkers.
    """
    @staticmethod
    def create_chunker(chunker_type, document_text, max_chunk_size=750, batch_size=50):
        if chunker_type == "agentic":
            return AgenticChunker(document_text, max_chunk_size, batch_size=25)
        elif chunker_type == "static":
            return StaticChunker(document_text, max_chunk_size, batch_size)
        elif chunker_type == "overlap":
            return OverlapChunker(document_text, max_chunk_size=5, batch_size=10)
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
